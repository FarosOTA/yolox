"""Export CVAT annotations and train as the user, optionally owning a temporary NFS mount."""

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid

from yolox.data.cvat import (
    RunReport,
    check_cache_space, check_local, check_network_hosts, class_merge_sources, cvat_hostname,
    export_project, prepare_dataset, read_export,
)
from yolox.exp import get_exp


def load_config(path):
    """Validate every table before any network, credential or mount operation."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        supplied = tomllib.load(stream)
    schema = {
        "cvat": {"host": (str,), "project_id": (int,)},
        "nfs": {"server": (str,), "export": (str,), "options": (list,), "frame_root": (str,)},
        "dataset": {"output_dir": (str,), "val_fraction": (int, float), "seed": (int,),
                    "validation_workers": (int,), "full_decode": (bool,)},
        "training": {"model": (str,), "exp_file": (str,), "devices": (int,), "batch": (int,),
                     "fp16": (bool,), "cache": (str,), "checkpoint": (str,),
                     "overrides": (dict,)},
        "classes": {"merge": (dict,), "exclude": (list,)},
    }
    defaults = {
        "cvat": {},
        "nfs": {"export": "/", "options": ["ro", "vers=4.2", "proto=tcp"]},
        "dataset": {"output_dir": "datasets/cvat-runs", "val_fraction": 0.2, "seed": 0,
                    "validation_workers": 16, "full_decode": False},
        "training": {"devices": 1, "batch": 8, "fp16": False, "cache": "disk",
                     "overrides": {}},
        "classes": {"merge": {}, "exclude": []},
    }
    unknown = set(supplied) - set(schema)
    if unknown:
        raise ValueError(f"Unknown config tables: {', '.join(sorted(unknown))}")
    for table, fields in schema.items():
        values = supplied.get(table, {})
        if not isinstance(values, dict):
            raise ValueError(f"{table} must be a TOML table")
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"Unknown keys in {table}: {', '.join(sorted(unknown))}; "
                             "PATs belong only in CVAT_ACCESS_TOKEN or the hidden prompt")
        for key, value in values.items():
            if type(value) not in fields[key] or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"Invalid value type or empty value for {table}.{key}")
        defaults[table].update(values)
    config = defaults
    cvat, nfs, dataset, training = (config[key] for key in ("cvat", "nfs", "dataset", "training"))
    if "host" not in cvat or "project_id" not in cvat or cvat["project_id"] < 1:
        raise ValueError("cvat.host and a positive cvat.project_id are required")
    cvat_hostname(cvat["host"])
    if "server" not in nfs:
        raise ValueError("nfs.server is required; there is no built-in server address")
    try:
        server = ipaddress.IPv4Address(nfs["server"])
    except ipaddress.AddressValueError:
        raise ValueError("nfs.server must be a LAN IPv4 address") from None
    if not server.is_private or server.is_loopback or server.is_link_local:
        raise ValueError("nfs.server must be a LAN IPv4 address")
    if not nfs["export"].startswith("/") or ":" in nfs["export"] or ".." in Path(nfs["export"]).parts:
        raise ValueError("nfs.export must be an absolute export path without traversal")
    options = nfs["options"]
    allowed = {"ro", "vers=4.2", "proto=tcp", "nosuid", "nodev", "noexec", "hard"}
    if (any(not isinstance(option, str) for option in options)
            or not {"ro", "vers=4.2", "proto=tcp"}.issubset(options)
            or set(options) - allowed or len(options) != len(set(options))):
        raise ValueError("nfs.options requires ro, vers=4.2, proto=tcp; "
                         "only nosuid, nodev, noexec and hard may be added")
    if not 0 < dataset["val_fraction"] < 1:
        raise ValueError("dataset.val_fraction must be between 0 and 1")
    if not 1 <= dataset["validation_workers"] <= 128:
        raise ValueError("dataset.validation_workers must be between 1 and 128")
    if training["devices"] < 1 or training["batch"] < 1:
        raise ValueError("training.devices and training.batch must be positive")
    if training["cache"] not in {"none", "ram", "disk"}:
        raise ValueError("training.cache must be none, ram or disk")
    if "model" in training and "exp_file" in training:
        raise ValueError("Choose training.model or training.exp_file, not both")
    if "exp_file" not in training:
        training.setdefault("model", "yolox-nano")
    for table, key in ((nfs, "frame_root"), (dataset, "output_dir"),
                       (training, "exp_file"), (training, "checkpoint")):
        if key in table:
            table[key] = str((path.parent / Path(table[key]).expanduser()).resolve())
    class_merge_sources(config["classes"]["merge"], config["classes"]["exclude"])
    exp = get_exp(training.get("exp_file"), training.get("model"))
    for key, value in training["overrides"].items():
        if key not in vars(exp):
            raise ValueError(f"Unknown training.overrides key: {key}")
        if key in {"data_dir", "train_ann", "val_ann", "num_classes", "output_dir"}:
            raise ValueError(f"training.overrides.{key} is managed by the workflow")
        if type(value) not in (str, int, float, bool, list):
            raise ValueError(f"Unsupported training.overrides value for {key}")
    exp.merge([part for key, value in training["overrides"].items() for part in (key, str(value))])
    return config


def training_arguments(config, output):
    training = config["training"]
    args = ["-f", training["exp_file"]] if "exp_file" in training else ["-n", training["model"]]
    args += ["-d", str(training["devices"]), "-b", str(training["batch"])]
    if training["fp16"]:
        args.append("--fp16")
    if training["cache"] != "none":
        args += ["--cache", training["cache"]]
    if "checkpoint" in training:
        args += ["-c", training["checkpoint"]]
    args += [part for key, value in training["overrides"].items() for part in (key, str(value))]
    return args + ["output_dir", str(output / "training")]


class TrainingInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        self.exit_code = 128 + signum


def interrupt(signum, frame):
    raise TrainingInterrupted(signum)


def stop_workers(process):
    """Terminate/reap the entire training session, including orphaned DataLoader workers."""
    if process is None:
        return True
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            # As a subreaper we inherit and can reap grandchildren after the trainer exits.
            try:
                while os.waitpid(-process.pid, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
    print(f"ERROR: training process group {process.pid} still has live workers; "
          "refusing to unmount under them", file=sys.stderr)
    return False


def cleanup_mount(path, workers_exited, source):
    """Never remove a mountpoint until unmount succeeds; report exact recovery commands."""
    detail = "Mount status could not be verified."
    try:
        mounted = subprocess.run(
            ["findmnt", "--noheadings", "--mountpoint", str(path), "--output", "SOURCE,TARGET"],
            capture_output=True, text=True,
        )
    except OSError as error:
        mounted = None
        detail = str(error)
    # A failed mount command can still have mounted something, so query even on that path.
    if mounted is not None and mounted.returncode == 1:
        path.rmdir()
        return True
    command = ["sudo", "umount", "--", str(path)]
    if mounted is not None and mounted.returncode == 0 and workers_exited:
        try:
            result = subprocess.run(["sudo", "-n", "umount", "--", str(path)],
                                    capture_output=True, text=True)
        except OSError as error:
            detail = str(error)
        else:
            if result.returncode == 0:
                path.rmdir()
                return True
            detail = result.stderr.strip() or "umount failed"
    print(f"ERROR: mount retained: {source} at {path}. "
          f"{detail}\n"
          f"After all training workers exit, run: {shlex.join(command)}\n"
          f"Then remove the directory: {shlex.join(['rmdir', '--', str(path)])}", file=sys.stderr)
    return False


def run_training(config):
    if sys.platform != "linux" or os.geteuid() == 0:
        raise ValueError("Run this Linux workflow as your normal user, never as root")
    report = RunReport({"cvat": config["cvat"], "nfs": config["nfs"],
                        "training": config["training"], "classes": config["classes"]})
    started = report.started
    cvat, nfs, dataset = (config[key] for key in ("cvat", "nfs", "dataset"))
    source = f"{nfs['server']}:{nfs['export']}"
    base = Path(dataset["output_dir"])
    output = base / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8])
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    mountpoint, process = None, None
    result = 1
    outcome = "failed"
    try:
        report.start_phase("Local and LAN preflight")
        print("Checking local storage and LAN connectivity...", flush=True)
        check_local(base)
        # Linux PR_SET_CHILD_SUBREAPER lets us reap orphaned DataLoader descendants before unmount.
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot enable training worker subreaper")
        check_network_hosts(cvat["host"], nfs["server"])
        # Export only annotations before mounting: capacity must fail without touching NFS images.
        with tempfile.TemporaryDirectory(prefix="yolox-cvat-") as temporary:
            archive = Path(temporary) / "annotations.zip"
            report.start_phase("CVAT export and download")
            export_project(cvat["host"], cvat["project_id"], archive)
            report.start_phase("Reading annotations")
            print("Reading exported annotations...", flush=True)
            data = read_export(archive)
            counts = Counter(annotation["category_id"] for annotation in data["annotations"])
            report.data["source"] = {
                "frames": len(data["images"]), "boxes": len(data["annotations"]),
                "classes": len(data["categories"]),
                "boxes_per_class": {category["name"]: counts[category["id"]]
                                    for category in data["categories"]},
            }
            print(f"Export contains {len(data['images'])} images, "
                  f"{len(data['annotations'])} boxes and {len(data['categories'])} classes", flush=True)
        report.start_phase("Experiment and cache capacity")
        training = config["training"]
        exp = get_exp(training.get("exp_file"), training.get("model"))
        exp.merge([part for key, value in training["overrides"].items()
                   for part in (key, str(value))])
        report.data["settings"]["effective"] = {
            "max_epoch": exp.max_epoch, "input_size": exp.input_size,
            "test_size": exp.test_size, "data_num_workers": exp.data_num_workers,
            "validation_workers": dataset["validation_workers"], "full_decode": dataset["full_decode"],
            "seed": dataset["seed"], "val_fraction": dataset["val_fraction"],
        }
        if training["cache"] == "disk":
            print("Checking disk cache capacity...", flush=True)
            report.data["cache_required_bytes"] = check_cache_space(
                data, output, exp.input_size, exp.test_size, dataset["val_fraction"], dataset["seed"])
        validation_token = uuid.uuid4().hex
        root = nfs.get("frame_root")
        if root is None:
            report.start_phase("NFS mount")
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, previous)
            try:
                mountpoint = Path(tempfile.mkdtemp(prefix="yolox-nfs-"))
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            root = mountpoint
            print("Mounting temporary NFS using sudo -n (no password prompt). "
                  "Authenticate with sudo -v in an interactive terminal first. Never run training as root.",
                  flush=True)
            pending = []
            for sig in previous:
                signal.signal(sig, lambda signum, frame: pending.append(signum))
            try:
                # Finish/reap the mount helper even if interrupted, then check and undo its result.
                subprocess.run([
                    "sudo", "-n", "mount", "-t", "nfs", "-o", ",".join(nfs["options"]),
                    source, str(mountpoint),
                ], check=True)
            finally:
                for sig in previous:
                    signal.signal(sig, interrupt)
                if pending:
                    raise TrainingInterrupted(pending[0])
        print("Preparing dataset: checking mount, validating images and mapping classes...", flush=True)
        prepare_dataset(cvat["host"], cvat["project_id"], root, output,
                        dataset["val_fraction"], dataset["seed"], merge=config["classes"]["merge"],
                        exclude=config["classes"]["exclude"],
                        nfs_server=nfs["server"], nfs_export=nfs["export"], data=data,
                        workers=dataset["validation_workers"], full_decode=dataset["full_decode"],
                        validation_token=validation_token, report=report)
        report.start_phase("Writing run configuration")
        training_args = training_arguments(config, output)
        resolved = {**config, "run": {"directory": str(output), "frame_root": str(root),
                    "training_arguments": training_args}}
        (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n",
                                                     encoding="utf-8")
        print(f"Run directory: {output}", flush=True)
        print(f"Preparation complete after {time.monotonic() - started:.1f}s; "
              "launching trainer (cache and epoch progress follow)...", flush=True)
        report.start_phase("Trainer (startup/cache/train/eval)")
        # Defer signals until the child handle is assigned, so cleanup cannot lose a spawned child.
        # Do not block them with sigprocmask here: that mask would survive exec in the trainer.
        pending = []
        for sig in previous:
            signal.signal(sig, lambda signum, frame: pending.append(signum))
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "yolox.tools.train", "--prepared-data",
                str(output), *training_args,
            ], start_new_session=True, restore_signals=True,
                env={**{key: value for key, value in os.environ.items() if key != "CVAT_ACCESS_TOKEN"},
                     "YOLOX_CVAT_VALIDATION": validation_token})
        finally:
            for sig in previous:
                signal.signal(sig, interrupt)
        if pending:
            raise TrainingInterrupted(pending[0])
        result = process.wait()
        report.data["trainer_exit_code"] = result
        outcome = "completed" if result == 0 else "failed"
        report.close_phase(outcome)
        print(f"Trainer exited with code {result} after {time.monotonic() - started:.1f}s total", flush=True)
    except BaseException as error:
        outcome = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report.data.update(error_type=type(error).__name__,
                           failed_phase=report.current["name"] if report.current else "initialization")
        report.close_phase(outcome)
        raise
    finally:
        # A second Ctrl-C/SIGTERM must not interrupt worker shutdown or leave a half-cleaned mount.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        report.start_phase("Worker and mount cleanup")
        cleanup_ok = False
        workers_exited = False
        mount_released = None
        try:
            print("Cleaning up training workers and any run-owned NFS mount...", flush=True)
            try:
                workers_exited = stop_workers(process)
            except OSError as error:
                print(f"ERROR: cannot confirm worker shutdown: {error}", file=sys.stderr)
                workers_exited = False
            if mountpoint is not None:
                mount_released = cleanup_mount(mountpoint, workers_exited, source)
                if not mount_released:
                    result = 1
            if not workers_exited:
                result = 1
            cleanup_ok = workers_exited and mount_released is not False
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            print(f"Run elapsed (including cleanup): {time.monotonic() - started:.1f}s", flush=True)
            report.data["cleanup"] = {"workers_exited": workers_exited,
                                      "owned_mount": mountpoint is not None, "unmounted": mount_released}
            report.close_phase("completed" if cleanup_ok else "failed")
            report.finish(outcome if cleanup_ok else "failed", output)
    return result if result >= 0 else 128 - result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Per-run TOML config (never include a PAT)")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        env_file = Path(args.config).resolve().parent / ".env"
        if env_file.is_file() and not os.environ.get("CVAT_ACCESS_TOKEN"):
            from dotenv import dotenv_values

            token = dotenv_values(env_file, interpolate=False).get("CVAT_ACCESS_TOKEN")
            if token:
                os.environ["CVAT_ACCESS_TOKEN"] = token
        return run_training(config)
    except TrainingInterrupted as error:
        return error.exit_code
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
