"""Preparation contracts with real images/COCO loaders and simulated Linux networking."""

import copy
from contextlib import nullcontext
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest

from yolox.data import cvat
from yolox.exp import get_exp


HOST = "https://cvat.example.com"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux LAN/mount workflow")


@pytest.fixture
def run_config(tmp_path):
    from tools.train_cvat import load_config
    path = tmp_path / "run.toml"
    path.write_text(f'[cvat]\nhost = "{HOST}"\nproject_id = 42\n'
                    '[nfs]\nserver = "192.0.2.10"\n[dataset]\noutput_dir = "runs"\n', encoding="utf-8")
    return load_config(path)


@pytest.fixture
def lan(tmp_path, monkeypatch):
    root = tmp_path / "frames"
    root.mkdir(parents=True)
    state = SimpleNamespace(
        root=root, addresses=["192.0.2.10"], route={"dev": "eth0"},
        speed="1000", duplex="full", source="192.0.2.10:/",
        fstype="nfs4", options="ro,vers=4.2,proto=tcp,addr=192.0.2.10",
        calls=[], connection_error=False,
    )
    # Reserved documentation addresses stand in for real RFC1918 LAN addresses only in tests.
    monkeypatch.setattr(cvat, "LAN_NETS", (*cvat.LAN_NETS, cvat.ipaddress.ip_network("192.0.2.0/24")))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM,
         6, "", (address, 0)) for address in state.addresses
    ])
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/sys/class/net/eth0/speed":
            return state.speed
        if str(path) == "/sys/class/net/eth0/duplex":
            return state.duplex
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    def run(args, **kwargs):
        state.calls.append(args)
        if args[:4] == ("ip", "-j", "route", "get"):
            data = [state.route]
        else:
            assert args[:3] == ("findmnt", "--json", "--target")
            network = Path(args[3]).is_relative_to(root)
            data = {"filesystems": [{
                "target": str(root) if network else str(tmp_path),
                "source": state.source if network else "/dev/nvme0n1p2",
                "fstype": state.fstype if network else "ext4",
                "options": state.options if network else "rw",
            }]}
        return subprocess.CompletedProcess(args, 0, json.dumps(data), "")

    monkeypatch.setattr(subprocess, "run", run)

    def connect(address, timeout):
        assert address == ("192.0.2.10", 2049) and timeout == 5
        if state.connection_error:
            raise OSError("connection refused")
        return nullcontext()

    monkeypatch.setattr(socket, "create_connection", connect)
    return state


def test_existing_nfs4_mount_and_direct_route(lan, capsys):
    assert cvat.check_network(HOST, lan.root, "192.0.2.10") == lan.root
    output = capsys.readouterr().out
    assert "1000 Mb/s, full duplex" in output
    assert "nfs4 192.0.2.10:/" in output
    assert "NFS TCP 192.0.2.10:2049 reachable" in output


@pytest.mark.parametrize("addresses,route,message", [
    (["100.64.0.1"], {"dev": "eth0"}, "Tailscale"),
    (["fd7a:115c:a1e0::123"], {"dev": "eth0"}, "Tailscale"),
    (["192.0.2.10"], {"dev": "tailscale0"}, "not on-link"),
    (["192.0.2.10"], {"dev": "eth0", "gateway": "192.0.2.1"}, "not on-link"),
    (["192.0.2.10"], {"dev": "lo", "type": "local"}, "not on-link"),
    (["192.0.2.10", "100.64.0.2"], {"dev": "eth0"}, "Tailscale"),
    (["127.0.0.1"], {"dev": "lo"}, "not a private LAN"),
])
def test_disallowed_network_paths(lan, addresses, route, message):
    lan.addresses, lan.route = addresses, route
    with pytest.raises(ValueError, match=message):
        cvat.check_network(HOST, lan.root, "192.0.2.10")


def test_public_dns_warns_possible_hairpin_and_fails(lan, monkeypatch):
    lan.addresses = ["203.0.113.10"]
    original = cvat.ipaddress.ip_address

    class PublicAddress(cvat.ipaddress.IPv4Address):
        is_global = True

    monkeypatch.setattr(cvat.ipaddress, "ip_address", lambda value: (
        PublicAddress(value) if value == "203.0.113.10" else original(value)
    ))
    with pytest.warns(UserWarning, match="possible NAT hairpinning.* /etc/hosts"):
        with pytest.raises(ValueError, match="not a private LAN"):
            cvat.check_network(HOST, lan.root, "192.0.2.10")


@pytest.mark.parametrize("speed,duplex", [("100", "full"), ("1000", "half")])
def test_slow_or_half_duplex_warns_but_does_not_block(lan, speed, duplex):
    lan.speed, lan.duplex = speed, duplex
    with pytest.warns(UserWarning, match="expected >=1000 Mb/s full duplex"):
        cvat.check_network(HOST, lan.root, "192.0.2.10")


def test_unknown_link_speed_warns(lan):
    lan.speed = "unknown"
    with pytest.warns(UserWarning, match="Cannot read speed/duplex"):
        cvat.check_network(HOST, lan.root, "192.0.2.10")


@pytest.mark.parametrize("fstype", ["ext4", "nfs", "cifs"])
def test_unmounted_or_non_nfs4_frames(lan, fstype):
    lan.fstype = fstype
    with pytest.raises(ValueError, match="existing nfs4 mount"):
        cvat.check_network(HOST, lan.root, "192.0.2.10")


@pytest.mark.parametrize("options,message", [
    ("rw,vers=4.2,proto=tcp,addr=192.0.2.10", "must be read-only"),
    ("ro,vers=4.2,proto=udp,addr=192.0.2.10", "must use proto=tcp"),
])
def test_mount_options_required(lan, options, message):
    lan.options = options
    with pytest.raises(ValueError, match=message):
        cvat.check_network(HOST, lan.root, "192.0.2.10")


@pytest.mark.parametrize("source", ["192.0.2.11:/", "192.0.2.10:/srv/frames"])
def test_wrong_nfs_host_or_export(lan, source):
    lan.source = source
    with pytest.raises(ValueError, match="Expected NFS source 192.0.2.10:/"):
        cvat.check_network(HOST, lan.root, "192.0.2.10")


def test_nfs_port_unreachable(lan):
    lan.connection_error = True
    with pytest.raises(ValueError, match="2049 is unreachable"):
        cvat.check_network(HOST, lan.root, "192.0.2.10")


def test_actual_mount_address_is_also_checked(lan, monkeypatch):
    original = socket.getaddrinfo
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *args, **kwargs: (
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.0.1", 0))]
        if host == "100.64.0.1" else original(host, *args, **kwargs)
    ))
    lan.options = "ro,proto=tcp,addr=100.64.0.1"
    with pytest.raises(ValueError, match="Tailscale"):
        cvat.check_network(HOST, lan.root, "192.0.2.10")


@pytest.fixture
def synthetic(lan):
    data = {"images": [], "annotations": [], "categories": [
        {"id": 9, "name": "car"}, {"id": 2, "name": "person"},
    ]}
    for index, ride in enumerate(("20260101_090000", "20260101_090000",
                                  "20260102_100000", "20260103_110000"), 1):
        name = f"{ride}/image{index}.jpg"
        path = lan.root / name
        path.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(path), np.full((12 + index, 20 + index, 3), index, np.uint8))
        data["images"].append({"id": index, "file_name": name,
                               "width": 20 + index, "height": 12 + index})
        data["annotations"].append({"id": index, "image_id": index, "category_id": 9,
                                    "bbox": [2, 3, 5, 4], "area": 20, "iscrowd": 0})
    # This unrelated invalid file must never be read or included.
    (lan.root / "ignored.jpg").touch()
    return data


@pytest.mark.parametrize("damage,message", [
    ("empty", "non-empty"), ("dimensions", "dimension mismatch"),
    ("missing", "No such file"), ("corrupt", "invalid image header"),
    ("truncated", "invalid image header"),
    ("absolute", "relative to frame root"), ("traversal", "relative to frame root"),
    ("video", "explicit mapping"), ("escape", "escapes frame root"),
])
def test_invalid_export_images_fail(lan, synthetic, tmp_path, damage, message):
    image = synthetic["images"][0]
    path = lan.root / image["file_name"]
    if damage == "empty":
        path.write_bytes(b"")
    elif damage == "dimensions":
        image["width"] += 1
    elif damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"not an image")
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:50])
    elif damage == "absolute":
        image["file_name"] = str(path)
    elif damage == "traversal":
        image["file_name"] = "ride/../../outside.jpg"
    elif damage == "video":
        image["file_name"] = "frame_000000.jpg"
    else:
        outside = tmp_path / "outside.jpg"
        path.rename(outside)
        path.symlink_to(outside)
    with pytest.raises(ValueError, match=message):
        cvat.validate_coco(synthetic, lan.root)


def test_permission_error_names_export_identity(lan, synthetic, monkeypatch):
    path = lan.root / synthetic["images"][0]["file_name"]
    original = Path.open

    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("Permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(ValueError, match="all_squash, anonuid=65534, anongid=65534"):
        cvat.validate_coco(synthetic, lan.root)


def install_export(monkeypatch, data, *, tasks=None):
    cvat_sdk = pytest.importorskip("cvat_sdk")
    from cvat_sdk.core.proxies.types import Location

    def make_client(host, *, access_token):
        assert host == HOST and access_token == "test-pat-not-a-real-secret"

        class Client:
            selected = data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            @property
            def projects(self):
                assert tasks is None, "Task selection must never export the entire project"
                return self

            @property
            def tasks(self):
                assert tasks is not None
                return self

            def retrieve(self, project_id):
                if tasks is None:
                    assert project_id == 42
                else:
                    self.selected = tasks[project_id]
                return self

            def export_dataset(self, format_name, destination, *, include_images, location, pbar):
                assert format_name == "COCO 1.0" and include_images is False
                assert location == Location.LOCAL
                assert destination.parent.is_dir()
                with zipfile.ZipFile(destination, "w") as bundle:
                    # Independent id spaces from two CVAT subsets.
                    data = self.selected
                    for index, images in enumerate((data["images"][:2], data["images"][2:])):
                        ids = {image["id"] for image in images}
                        subset = {**data, "images": copy.deepcopy(images), "annotations": [
                            copy.deepcopy(a) for a in data["annotations"] if a["image_id"] in ids
                        ]}
                        remap = {image["id"]: n for n, image in enumerate(subset["images"], 1)}
                        for image in subset["images"]:
                            image["id"] = remap[image["id"]]
                        for a in subset["annotations"]:
                            a["image_id"] = remap[a["image_id"]]
                        bundle.writestr(f"annotations/instances_{index}.json", json.dumps(subset))
                size = destination.stat().st_size
                with pbar.task(total=size, desc="Downloading", unit="B", unit_scale=True):
                    pbar.advance(size)
        return Client()

    monkeypatch.setattr(cvat_sdk, "make_client", make_client)
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "test-pat-not-a-real-secret")


def test_prepare_preflight_and_real_training_loaders(lan, synthetic, tmp_path, monkeypatch, capsys):
    install_export(monkeypatch, synthetic)
    output = tmp_path / "prepared"
    manifest = cvat.prepare_dataset(HOST, 42, lan.root, output, seed=13,
                                    nfs_server="192.0.2.10")
    assert manifest["num_classes"] == 2
    assert manifest["val_rides"] == ["20260101_090000"]
    assert json.loads((output / "classes.json").read_text()) == ["person", "car"]
    exp = get_exp(exp_name="yolox-nano")
    exp.input_size = exp.test_size = (32, 32)
    cvat.configure_training(exp, output)
    assert exp.num_classes == 2 and exp.data_dir == str(output)
    train = exp.get_dataset(cache=True, cache_type="disk")
    val = exp.get_eval_dataset()
    assert len(train) == 2 and len(val) == 1
    assert manifest["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert {image["file_name"] for image in train.coco.dataset["images"]}.isdisjoint(
        image["file_name"] for image in val.coco.dataset["images"])
    assert train.class_ids == val.class_ids == [2, 9]
    assert train.pull_item(0)[1][0, 4] == 1  # category 9 is index 1, not export order 0.
    assert val.load_image(0).shape[:2] == (14, 22)
    assert train.read_img(0).shape[:2] == (19, 32)  # 13 × (32/21), floored by YOLOX.
    cvat.configure_training(exp, output, evaluation_split="test")
    test = exp.get_eval_dataset()
    assert len(test) == 1 and test.load_image(0).shape[:2] == (16, 24)
    assert test.coco.dataset["images"][0]["file_name"] == synthetic["images"][3]["file_name"]
    assert list((output / "cache_train2017").rglob("*.npy"))
    assert not list(lan.root.rglob("*.npy"))
    assert len(list(lan.root.rglob("*.jpg"))) == 5
    assert cvat.preflight_dataset(output) == manifest
    assert "test-pat-not-a-real-secret" not in capsys.readouterr().out
    # Revalidation warns rather than aborting if a source frame has become unreadable.
    (lan.root / synthetic["images"][0]["file_name"]).write_bytes(b"")
    cvat.configure_training(exp, output)
    from yolox.data.image_errors import image_errors
    assert any("non-empty" in record["reason"] for record in image_errors(output))


def test_cache_symlink_rejected(lan, synthetic, tmp_path, monkeypatch):
    install_export(monkeypatch, synthetic)
    output = tmp_path / "prepared"
    cvat.prepare_dataset(HOST, 42, lan.root, output, nfs_server="192.0.2.10")
    (output / "cache_train2017").symlink_to(lan.root, target_is_directory=True)
    with pytest.raises(ValueError, match="Cache must not contain symlinks"):
        cvat.preflight_dataset(output)


def test_preparation_skips_bad_frames_and_reports_at_run_end(lan, synthetic, tmp_path):
    from loguru import logger
    from yolox.data.image_errors import record_image_error

    bad = synthetic["images"][0]
    (lan.root / bad["file_name"]).unlink()
    output = tmp_path / "prepared"
    report = cvat.RunReport({})
    report.data["source"] = {"frames": 4, "boxes": 4, "classes": 2,
                              "boxes_per_class": {"car": 4, "person": 0}}
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        manifest = cvat.prepare_dataset(HOST, 42, lan.root, output, data=synthetic,
                                        nfs_server="192.0.2.10", report=report)
        assert sum(manifest["split_counts"].values()) == 3
        for filename in cvat.SPLIT_FILES.values():
            data = json.loads((output / "annotations" / filename).read_text())
            assert all(image["id"] != bad["id"] for image in data["images"])
            assert all(a["image_id"] != bad["id"] for a in data["annotations"])
        assert "End of image validation" in "".join(messages)
        assert bad["file_name"] in "".join(messages)
        # A diagnostic written by the trainer must survive into the supervisor summary.
        record_image_error(output, {"split": "train2017", "image_id": 2,
                                    "path": "ride/later.jpg", "reason": "OpenCV returned no pixels"})
        report.data["cleanup"] = {}
        messages.clear()
        report.finish("completed", output)
        summary = json.loads((output / "summary.json").read_text())
        assert len(summary["skipped_frames"]) == 2
        assert "End of training run" in "".join(messages)
        assert bad["file_name"] in "".join(messages) and "ride/later.jpg" in "".join(messages)
    finally:
        logger.remove(sink)


@pytest.mark.parametrize("cache", [None, "ram", "disk", "partial_disk"])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "dimensions"])
def test_unreadable_coco_frame_skips_with_matching_labels(lan, synthetic, tmp_path, cache, damage):
    from torch.utils.data import DataLoader
    from yolox.data import ValTransform
    from yolox.data.datasets import COCODataset
    from yolox.data.image_errors import image_errors

    # Validation succeeds, then the source changes before cache construction/loading.
    cvat.validate_coco(synthetic, lan.root)
    output = tmp_path / "dataset"
    (output / "annotations").mkdir(parents=True)
    (output / "annotations" / cvat.TRAIN_JSON).write_text(json.dumps(synthetic))
    (output / "train2017").symlink_to(lan.root, target_is_directory=True)
    bad = synthetic["images"][0]
    path = lan.root / bad["file_name"]
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"not an image")
    else:
        assert cv2.imwrite(str(path), np.zeros((7, 9, 3), np.uint8))
    if cache == "partial_disk":
        (output / "cache_train2017").mkdir()
    dataset = COCODataset(str(output), img_size=(32, 32), preproc=ValTransform(),
                          cache=cache is not None, cache_type="disk" if cache == "partial_disk" else cache)
    # Real worker processes exercise error propagation and persistent diagnostics.
    batches = list(DataLoader(dataset, batch_size=2, num_workers=2))
    assert [int(i) for batch in batches for i in batch[3]] == [2, 2, 3, 4]
    pixels, labels, dimensions, ids = dataset.pull_item(0)
    assert ids.tolist() == [2] and dimensions == (14, 22)
    assert np.all(pixels == 2)
    np.testing.assert_allclose(labels[0], [2 * 32 / 22, 3 * 32 / 22, 7 * 32 / 22, 7 * 32 / 22, 1])
    records = image_errors(output)
    assert len(records) == 1 and records[0]["image_id"] == 1
    assert records[0]["path"] == str(output / "train2017" / bad["file_name"])
    # Complete source loss must terminate instead of retrying forever.
    for image in synthetic["images"]:
        (lan.root / image["file_name"]).unlink(missing_ok=True)
    uncached = COCODataset(str(output), img_size=(32, 32))
    with pytest.raises(RuntimeError, match="No readable frames"):
        uncached.pull_item(0)


def test_coco_metrics_exclude_skips_but_include_readable_misses(lan, synthetic, tmp_path, monkeypatch):
    from loguru import logger
    import torch
    from torch.utils.data import DataLoader
    from yolox.data import ValTransform
    from yolox.data.datasets import COCODataset
    from yolox.evaluators.coco_evaluator import COCOEvaluator

    output = tmp_path / "dataset"
    (output / "annotations").mkdir(parents=True)
    (output / "annotations" / cvat.TRAIN_JSON).write_text(json.dumps(synthetic))
    (output / "train2017").symlink_to(lan.root, target_is_directory=True)
    (lan.root / synthetic["images"][0]["file_name"]).unlink()
    dataset = COCODataset(str(output), img_size=(32, 32), preproc=ValTransform())
    evaluator = COCOEvaluator(DataLoader(dataset, batch_size=2), (32, 32), 0.01, 0.65, 2)
    monkeypatch.setattr("yolox.evaluators.coco_evaluator.tempfile.tempdir", str(tmp_path))
    # Run the real evaluation loop/NMS/COCO metrics with CPU fixture predictions;
    # this is not a CUDA hardware test.
    monkeypatch.setattr(torch.cuda, "FloatTensor", torch.FloatTensor)
    monkeypatch.setattr("yolox.evaluators.coco_evaluator.time_synchronized", cvat.time.time)

    class FixtureDetector(torch.nn.Module):
        def forward(self, images):
            ids = images[:, 0, 0, 0]
            result = torch.zeros((len(images), 2, 7))
            result[:, 0, :4] = torch.tensor([4.5, 5, 5, 4]) * (32 / (20 + ids[:, None]))
            result[:, 0, 4] = (ids < 4).float()
            result[:, 0, 6] = 1
            return result

    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        (ap, _, _), predictions = evaluator.evaluate(FixtureDetector(), return_outputs=True)
    finally:
        logger.remove(sink)
    # 2/3 recall: 67 of COCO's 101 recall thresholds. Frame 1 was skipped;
    # frame 4 was read successfully but missed, and must still reduce AP.
    assert ap == pytest.approx(67 / 101)
    assert set(predictions) == {2, 3, 4} and predictions[4]["bboxes"] == []
    assert "End of validation" in "".join(messages)
    assert synthetic["images"][0]["file_name"] in "".join(messages)


@pytest.mark.parametrize("ride_sizes,expected", [
    ([3], {"train": 1, "val": 1, "test": 1}),
    ([10], {"train": 7, "val": 2, "test": 1}),
    ([20], {"train": 14, "val": 3, "test": 3}),
    ([22, 58, 1150, 1309], {"train": 1777, "val": 381, "test": 381}),
])
def test_prepared_image_split_ratios_and_contiguous_segments(lan, synthetic, tmp_path,
                                                            ride_sizes, expected):
    source = {"images": [], "annotations": [], "categories": synthetic["categories"]}
    original = lan.root / synthetic["images"][0]["file_name"]
    for ride, size in enumerate(ride_sizes):
        folder = lan.root / f"ride-{ride}"
        folder.mkdir()
        for frame in range(size):
            path = folder / f"{frame:06d}.jpg"
            path.hardlink_to(original)
            image_id = len(source["images"]) + 1
            source["images"].append({"id": image_id, "file_name": path.relative_to(lan.root).as_posix(),
                                      "width": 21, "height": 13})
            source["annotations"].append({**synthetic["annotations"][0], "id": image_id,
                                          "image_id": image_id})
    source["images"].reverse()  # Export order must not change membership.
    output = tmp_path / "split"
    manifest = cvat.prepare_dataset(HOST, 42, lan.root, output, data=source,
                                    nfs_server="192.0.2.10", seed=7)
    assert manifest["split_counts"] == expected
    assert manifest["split_ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    subsets = {split: json.loads((output / "annotations" / filename).read_text())
               for split, filename in cvat.SPLIT_FILES.items()}
    names = []
    for split, subset in subsets.items():
        assert len(subset["images"]) == expected[split]
        assert {a["image_id"] for a in subset["annotations"]} == {i["id"] for i in subset["images"]}
        names.extend(image["file_name"] for image in subset["images"])
        for ride in range(len(ride_sizes)):
            frames = sorted(int(Path(image["file_name"]).stem) for image in subset["images"]
                            if image["file_name"].startswith(f"ride-{ride}/"))
            if frames:
                assert frames == list(range(frames[0], frames[-1] + 1))
    assert len(names) == len(set(names)) == sum(ride_sizes)
    assert set(names) == {image["file_name"] for image in source["images"]}
    reordered = cvat.split_dataset({"images": list(reversed(source["images"]))}, seed=7)
    for split in subsets:
        assert {image["file_name"] for image in reordered[split]} == {
            image["file_name"] for image in subsets[split]["images"]}
    assert cvat.preflight_dataset(output) == manifest


def test_preflight_rejects_overlapping_test_images(lan, synthetic, tmp_path):
    output = tmp_path / "overlap"
    cvat.prepare_dataset(HOST, 42, lan.root, output, data=synthetic, nfs_server="192.0.2.10")
    (output / "annotations" / cvat.TEST_JSON).write_text(
        (output / "annotations" / cvat.VAL_JSON).read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="images overlap"):
        cvat.preflight_dataset(output)


def test_sdk_failure_does_not_expose_pat(tmp_path, monkeypatch, capsys):
    cvat_sdk = pytest.importorskip("cvat_sdk")
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "test-pat-not-a-real-secret")

    def fail(*args, **kwargs):
        raise RuntimeError("Request Authorization: Bearer test-pat-not-a-real-secret")

    monkeypatch.setattr(cvat_sdk, "make_client", fail)
    with pytest.raises(ValueError, match="CVAT export failed") as error:
        cvat.export_project(HOST, 42, tmp_path / "export.zip")
    import traceback
    assert "test-pat-not-a-real-secret" not in "".join(traceback.format_exception(error.value))
    assert "test-pat-not-a-real-secret" not in str(capsys.readouterr())
    assert not list(tmp_path.iterdir())


def test_prompt_used_when_environment_missing(lan, synthetic, tmp_path, monkeypatch):
    install_export(monkeypatch, synthetic)
    monkeypatch.delenv("CVAT_ACCESS_TOKEN")
    monkeypatch.setattr(cvat.getpass, "getpass", lambda prompt: "test-pat-not-a-real-secret")
    output = tmp_path / "prepared"
    cvat.prepare_dataset(HOST, 42, lan.root, output, nfs_server="192.0.2.10")
    assert not any("test-pat-not-a-real-secret" in path.read_text()
                   for path in output.rglob("*.json"))


def test_output_on_share_and_too_few_images_rejected(lan, synthetic, tmp_path, monkeypatch):
    install_export(monkeypatch, synthetic)
    with pytest.raises(ValueError, match="local filesystem"):
        cvat.prepare_dataset(HOST, 42, lan.root, lan.root / "prepared", nfs_server="192.0.2.10")
    synthetic["images"] = synthetic["images"][:2]
    synthetic["annotations"] = synthetic["annotations"][:2]
    with pytest.raises(ValueError, match="At least three images"):
        cvat.prepare_dataset(HOST, 42, lan.root, tmp_path / "prepared", nfs_server="192.0.2.10")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("failure", [
    None, "network", "mount", "mount-partial", "mount-sigterm", "preflight", "exception",
    "ctrl-c", "sigterm", "spawn-sigterm", "umount", "umount-oserror", "workers", "trainer-exit",
])
def test_supervised_mount_lifecycle(tmp_path, monkeypatch, capsys, failure, run_config, synthetic):
    import os
    import signal
    from tools import train_cvat

    events = []
    mountpoint = tmp_path / "temporary-mount"
    state = {"mounted": False, "group": False}
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train_cvat.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=lambda *a: 0))
    monkeypatch.setattr(train_cvat, "check_local", lambda path: None)
    monkeypatch.setattr(train_cvat, "export_project", lambda *args: None)
    monkeypatch.setattr(train_cvat, "read_export", lambda *args: synthetic)
    monkeypatch.setattr(train_cvat, "check_cache_space", lambda *args: None)

    original_mkdtemp = train_cvat.tempfile.mkdtemp

    def temporary(*args, **kwargs):
        if args:  # The annotation TemporaryDirectory is separate from the owned mountpoint.
            return original_mkdtemp(*args, **kwargs)
        mountpoint.mkdir()
        return str(mountpoint)

    monkeypatch.setattr(train_cvat.tempfile, "mkdtemp", temporary)

    def network(host, server):
        assert host == HOST and server == "192.0.2.10"
        events.append("network")
        if failure == "network":
            raise ValueError("network failed")

    def prepare(*args, **kwargs):
        assert state["mounted"]
        events.append("preflight")
        if failure == "preflight":
            raise ValueError("image preflight failed")
        args[3].mkdir(parents=True)

    monkeypatch.setattr(train_cvat, "check_network_hosts", network)
    monkeypatch.setattr(train_cvat, "prepare_dataset", prepare)

    def run(command, **kwargs):
        if command[:3] == ["sudo", "-n", "mount"]:
            assert events == ["network"]
            assert command[3:] == ["-t", "nfs", "-o", "ro,vers=4.2,proto=tcp",
                                   "192.0.2.10:/", str(mountpoint)]
            events.append("mount")
            state["mounted"] = failure != "mount"
            if failure in {"mount", "mount-partial"}:
                raise subprocess.CalledProcessError(1, command)
            if failure == "mount-sigterm":
                signal.raise_signal(signal.SIGTERM)
        elif command[0] == "findmnt":
            return subprocess.CompletedProcess(command, 0 if state["mounted"] else 1,
                                               "192.0.2.10:/ " + str(mountpoint), "")
        else:
            assert command == ["sudo", "-n", "umount", "--", str(mountpoint)]
            assert not state["group"], "unmount must wait for workers"
            events.append("umount")
            if failure == "umount":
                return subprocess.CompletedProcess(command, 1, "", "target is busy")
            if failure == "umount-oserror":
                raise OSError("sudo executable unavailable")
            state["mounted"] = False
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    class Process:
        pid = 12345

        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"]
            assert command[:4] == [sys.executable, "-m", "yolox.tools.train", "--prepared-data"]
            events.append("train")
            state["group"] = True
            if failure == "spawn-sigterm":
                signal.raise_signal(signal.SIGTERM)

        def wait(self):
            if failure == "exception":
                raise RuntimeError("trainer failed")
            if failure in {"ctrl-c", "sigterm"}:
                signal.raise_signal(signal.SIGINT if failure == "ctrl-c" else signal.SIGTERM)
            return 3 if failure == "trainer-exit" else 0

    monkeypatch.setattr(subprocess, "Popen", Process)

    def stop(process):
        if process is not None:
            events.append("workers-exited" if failure != "workers" else "workers-stuck")
        state["group"] = failure == "workers"
        return not state["group"]

    monkeypatch.setattr(train_cvat, "stop_workers", stop)
    if failure in {"network", "mount", "mount-partial", "mount-sigterm", "preflight", "exception",
                   "ctrl-c", "sigterm", "spawn-sigterm"}:
        with pytest.raises((ValueError, subprocess.CalledProcessError, RuntimeError, KeyboardInterrupt)):
            train_cvat.run_training(run_config)
    else:
        expected = 3 if failure == "trainer-exit" else 1 if failure in {"umount", "umount-oserror", "workers"} else 0
        assert train_cvat.run_training(run_config) == expected
    captured = capsys.readouterr()
    status = ("completed" if failure is None else "interrupted"
              if failure in {"ctrl-c", "sigterm", "mount-sigterm", "spawn-sigterm"} else "failed")
    assert f"Run summary — {status}" in captured.out
    assert "Total elapsed (including cleanup):" in captured.out
    for summary_path in Path(run_config["dataset"]["output_dir"]).glob("*/summary.json"):
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == status
        assert summary["phases"][-1]["name"] == "Worker and mount cleanup"
        assert sum(phase["seconds"] for phase in summary["phases"]) + summary["overhead_seconds"] == pytest.approx(summary["total_seconds"])
    if failure in {"umount", "umount-oserror", "workers"}:
        assert state["mounted"] and mountpoint.exists()
        message = captured.err
        assert f"192.0.2.10:/ at {mountpoint}" in message
        assert f"sudo umount -- {mountpoint}" in message
        assert f"rmdir -- {mountpoint}" in message
    else:
        assert not state["mounted"] and not mountpoint.exists()
    if "train" in events and failure not in {"workers"}:
        assert events.index("workers-exited") < events.index("umount")


def test_supervisor_never_unmounts_existing_root(lan, synthetic, monkeypatch, run_config):
    from tools import train_cvat
    import os
    install_export(monkeypatch, synthetic)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train_cvat.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=lambda *a: 0))
    run_config["nfs"]["frame_root"] = str(lan.root)
    # Fails after the existing-root preflight/export, without issuing any mount or unmount command.
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(OSError("no trainer")))
    with pytest.raises(OSError, match="no trainer"):
        train_cvat.run_training(run_config)
    assert lan.root.exists()
    assert all(call[0] in {"findmnt", "ip"} for call in lan.calls)


@pytest.mark.parametrize("extra,message", [
    ('[typo]\nfoo = 1', "Unknown config tables: typo"),
    ('[training]\nbtach = 4', "Unknown keys in training: btach"),
    ('[training.overrides]\nmax_epcoh = 3', "Unknown training.overrides key: max_epcoh"),
    ('[training.overrides]\noutput_dir = "remote"', "output_dir is managed"),
    ('[nfs]\nserver = "192.0.2.10"\noptions = ["rw", "vers=4.2", "proto=tcp"]', "nfs.options requires ro"),
    ('[training]\nfp16 = "false"', "Invalid value type.*training.fp16"),
    ('[training]\nmodel = "yolox-nano"\nexp_file = "exp.py"', "not both"),
    ('[training]\nresume = true', "Unknown keys in training: resume"),
    ('[training]\ncheckpoint = "weights.pth"\nresume = false', "Unknown keys in training: resume"),
    ('[dataset]\nval_fraction = 0.2', "Unknown keys in dataset: val_fraction"),
    ('[training.overrides]\ntest_ann = "custom.json"', "test_ann is managed"),
    ('[classes.merge]\nvehicle = []', "non-empty array"),
    ('[classes.merge]\nvehicle = "car"', "non-empty array"),
    ('[classes.merge]\nvehicle = ["car"]\nother = ["car"]', "'car'.*'vehicle'.*'other'"),
    ('[classes]\nexclude = "car"', "Invalid value type.*classes.exclude"),
    ('[classes]\nexclude = [2]', "unique non-empty source label names"),
    ('[classes]\nexclude = ["car", "car"]', "unique non-empty source label names"),
    ('[classes]\nexclude = [" "]', "unique non-empty source label names"),
    ('[classes]\nexclude = ["car"]\n[classes.merge]\nvehicle = ["car"]', "overlaps merge sources or targets: car"),
    ('[classes]\nexclude = ["vehicle"]\n[classes.merge]\nvehicle = ["car"]', "overlaps merge sources or targets: vehicle"),
])
def test_config_rejects_typos_and_invalid_settings(tmp_path, extra, message):
    from tools.train_cvat import load_config
    path = tmp_path / "bad.toml"
    path.write_text(f'[cvat]\nhost = "{HOST}"\nproject_id = 42\n'
                    + ('' if '[nfs]' in extra else '[nfs]\nserver = "192.0.2.10"\n')
                    + extra, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_config_token_rejected_without_echo(tmp_path, monkeypatch, capsys):
    from tools import train_cvat
    path = tmp_path / "secret.toml"
    path.write_text(f'[cvat]\nhost = "{HOST}"\nproject_id = 42\n'
                    'token = "test-pat-not-a-real-secret"\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_cvat.py", str(path)])
    assert train_cvat.main() == 1
    output = capsys.readouterr()
    assert "Unknown keys in cvat: token" in output.err
    assert "test-pat-not-a-real-secret" not in output.err + output.out


@pytest.mark.parametrize("missing", ["host", "project_id", "server"])
def test_config_requires_deployment_settings(tmp_path, missing):
    from tools.train_cvat import load_config
    fields = {"host": f'host = "{HOST}"\n', "project_id": 'project_id = 42\n',
              "server": 'server = "192.0.2.10"\n'}
    fields[missing] = ""
    path = tmp_path / "missing.toml"
    path.write_text('[cvat]\n' + fields["host"] + fields["project_id"]
                    + '[nfs]\n' + fields["server"], encoding="utf-8")
    with pytest.raises(ValueError, match="required"):
        load_config(path)


@pytest.mark.parametrize("selection", [
    "task_ids = []", "task_ids = [0]", "task_ids = [true]", "task_ids = [101, 101]",
    'task_ids = ["101"]', "task_ids = [[101]]", "task_ids = [101]\nproject_id = 42",
])
def test_config_rejects_invalid_task_selection(tmp_path, selection):
    from tools.train_cvat import load_config
    path = tmp_path / "tasks.toml"
    path.write_text(f'[cvat]\nhost = "{HOST}"\n{selection}\n'
                    '[nfs]\nserver = "192.0.2.10"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cvat.task_ids"):
        load_config(path)


def test_config_paths_and_training_settings(tmp_path):
    from tools.train_cvat import load_config, training_arguments
    from tools.train import make_parser
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "train.toml"
    path.write_text(f'[cvat]\nhost = "{HOST}"\nproject_id = 42\n'
                    '[nfs]\nserver = "192.0.2.11"\nexport = "/frames"\n'
                    'frame_root = "../mounted"\n[dataset]\noutput_dir = "../runs"\n'
                    '[training]\nmodel = "yolox-s"\ndevices = 2\nbatch = 6\n'
                    'fp16 = true\ncache = "ram"\ncheckpoint = "../previous.pth"\n'
                    '[training.overrides]\nmax_epoch = 7\ninput_size = [96, 64]\n'
                    '[classes]\nexclude = ["cone"]\n', encoding="utf-8")
    config = load_config(path)
    output = tmp_path / "runs" / "new"
    parsed = make_parser().parse_args(training_arguments(config, output))
    assert config["nfs"]["frame_root"] == str(tmp_path / "mounted")
    assert config["dataset"]["output_dir"] == str(tmp_path / "runs")
    assert config["classes"]["exclude"] == ["cone"]
    assert (parsed.name, parsed.devices, parsed.batch_size) == ("yolox-s", 2, 6)
    assert parsed.fp16 and not parsed.resume and parsed.cache == "ram"
    assert parsed.ckpt == str(tmp_path / "previous.pth")
    exp = get_exp(exp_name=parsed.name)
    exp.merge(parsed.opts)
    assert exp.max_epoch == 7 and exp.input_size == (96, 64)
    assert exp.output_dir == str(output / "training")
    # The checked-in example is executable config, not an out-of-date documentation sketch.
    repo = Path(__file__).parents[1]
    example = load_config(repo / "train.toml.example")
    assert example["training"]["model"] == "yolox-nano"
    assert example["dataset"]["output_dir"] == str(repo / "datasets/cvat-runs")
    parsed_example = make_parser().parse_args(training_arguments(example, output))
    assert parsed_example.ckpt is None and not parsed_example.resume
    assert parsed_example.devices == 1 and parsed_example.batch_size == 8
    assert parsed_example.cache == "disk" and not parsed_example.fp16
    # The alternative experiment-file setting follows the same parser/override contract.
    experiment = Path(__file__).parents[1] / "exps/default/yolox_nano.py"
    path.write_text(path.read_text().replace('model = "yolox-s"',
                                            f'exp_file = "{experiment.as_posix()}"'),
                    encoding="utf-8")
    parsed = make_parser().parse_args(training_arguments(load_config(path), output))
    assert parsed.exp_file == str(experiment) and parsed.name is None
    exp = get_exp(parsed.exp_file, parsed.name)
    exp.merge(parsed.opts)
    assert exp.max_epoch == 7 and exp.num_classes == 80


@pytest.fixture
def grouped_export(synthetic):
    categories = [(15, "vehicle"), (4, "truck"), (3, "person"), (1, "car"),
                  (9, "motorcycle"), (11, "cone"), (7, "bicycle"), (12, "bus")]
    synthetic["categories"] = [{"id": id_, "name": name} for id_, name in categories]
    synthetic["annotations"] = [
        {"id": image["id"] * 100 + id_, "image_id": image["id"], "category_id": id_,
         "bbox": [2, 3, 5, 4], "area": 20, "iscrowd": 0}
        for image in synthetic["images"] for id_, _ in sorted(categories)
    ]
    return synthetic


@pytest.mark.parametrize("exclude", [[], ["person", "cone"]])
@pytest.mark.parametrize("evaluation_split", ["val", "test"])
def test_grouped_merge_run_records_loaders_and_evaluation(lan, grouped_export, run_config,
                                                         monkeypatch, capsys, tmp_path, exclude,
                                                         evaluation_split):
    from tools import train_cvat
    from yolox.evaluators.coco_evaluator import COCOEvaluator
    import os
    install_export(monkeypatch, grouped_export)
    run_config["nfs"]["frame_root"] = str(lan.root)
    run_config["classes"]["merge"] = {
        "two_wheeler": ["motorcycle", "bicycle"],
        "vehicle": ["bus", "truck", "vehicle", "car"],
    }
    run_config["classes"]["exclude"] = exclude
    expected_classes = ["vehicle", "two_wheeler"] if exclude else ["vehicle", "person", "two_wheeler", "cone"]
    expected_mapping = ([
        ("car", 0), ("person", None), ("truck", 0), ("bicycle", 1),
        ("motorcycle", 1), ("cone", None), ("bus", 0), ("vehicle", 0),
    ] if exclude else [
        ("car", 0), ("person", 1), ("truck", 0), ("bicycle", 2),
        ("motorcycle", 2), ("cone", 3), ("bus", 0), ("vehicle", 0),
    ])
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train_cvat.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=lambda *a: 0))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: SimpleNamespace(wait=lambda: 0))
    monkeypatch.setattr(train_cvat, "stop_workers", lambda process: True)
    for _ in range(2):
        assert train_cvat.run_training(run_config) == 0
    runs = sorted(Path(run_config["dataset"]["output_dir"]).iterdir())
    assert len(runs) == 2 and runs[0] != runs[1]
    for output in runs:
        assert json.loads((output / "classes.json").read_text()) == expected_classes
        mapping = json.loads((output / "class_mapping.json").read_text())
        assert [(entry["source"], entry["class_index"]) for entry in mapping] == expected_mapping
        for entry in mapping:
            if entry["source"] in exclude:
                assert entry["class"] is None and entry["category_id"] is None
        resolved = json.loads((output / "resolved_config.json").read_text())
        assert resolved["classes"] == run_config["classes"]
        assert resolved["run"]["directory"] == str(output)
        assert resolved["run"]["training_arguments"][-2:] == ["output_dir", str(output / "training")]
        assert "test-pat-not-a-real-secret" not in (output / "resolved_config.json").read_text()
    exp = get_exp(exp_name="yolox-nano")
    exp.input_size = exp.test_size = (32, 32)
    cvat.configure_training(exp, runs[0], evaluation_split=evaluation_split)
    assert exp.num_classes == len(expected_classes)
    train, val = exp.get_dataset(), exp.get_eval_dataset()
    assert train._classes == val._classes == tuple(expected_classes)
    for dataset in (train, val):
        assert dataset.class_ids == ([1, 2] if exclude else [1, 2, 3, 4])
        assert dataset.pull_item(0)[1][:, 4].tolist() == ([0, 0, 1, 1, 0, 0] if exclude else [0, 1, 0, 2, 2, 3, 0, 0])
        assert len(dataset.coco.dataset["annotations"]) == len(dataset) * (6 if exclude else 8)
    evaluator = COCOEvaluator(SimpleNamespace(dataset=val, batch_size=1), (32, 32), 0.01, 0.65, len(expected_classes))
    # Scope the evaluator's temporary predictions to this test's filesystem.
    monkeypatch.setattr("yolox.evaluators.coco_evaluator.tempfile.mkstemp",
                        lambda: (0, str(tmp_path / "predictions.json")))
    predictions = [{"image_id": a["image_id"], "category_id": a["category_id"],
                    "bbox": a["bbox"], "score": 1.0} for a in val.coco.dataset["annotations"]]
    ap, _, report = evaluator.evaluate_prediction(predictions, np.array([0.1, 0.1, 1]))
    assert ap == pytest.approx(1.0)
    for name in expected_classes:
        assert name in report
    for name in ["car", "truck", "bus", "bicycle", "motorcycle", *exclude]:
        assert f"| {name} " not in report
    assert "Final source-to-class mapping" in capsys.readouterr().out


@pytest.mark.parametrize("merge,exclude,message", [
    ({"vehicle": ["car", "typo"]}, [], "Unknown CVAT labels.*typo"),
    ({"vehicle": ["car"], "other": ["car"]}, [], "'car'.*'vehicle'.*'other'"),
    ({"vehicle": []}, [], "non-empty array"),
    ({"vehicle": ["car"]}, [], "'vehicle'.*existing CVAT label"),
    ({}, ["typo"], "Unknown CVAT labels in classes.exclude: typo"),
    ({}, ["car", "person", "truck", "bicycle", "motorcycle", "cone", "bus", "vehicle"], "leave at least one class"),
])
def test_invalid_class_groups_fail_before_prepared_output(lan, grouped_export, tmp_path,
                                                         monkeypatch, merge, exclude, message):
    install_export(monkeypatch, grouped_export)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=message):
        cvat.prepare_dataset(HOST, 42, lan.root, output, merge=merge, exclude=exclude, nfs_server="192.0.2.10")
    assert not output.exists()


def test_exclusion_without_merges_keeps_empty_images(lan, synthetic, tmp_path, monkeypatch):
    synthetic["annotations"][0]["category_id"] = 2  # Only the first image retains a box.
    install_export(monkeypatch, synthetic)
    output = tmp_path / "excluded"
    cvat.prepare_dataset(HOST, 42, lan.root, output, exclude=["car"], nfs_server="192.0.2.10")
    exp = get_exp(exp_name="yolox-nano")
    cvat.configure_training(exp, output)
    assert exp.num_classes == 1
    datasets = [exp.get_dataset(), exp.get_eval_dataset()]
    cvat.configure_training(exp, output, evaluation_split="test")
    datasets.append(exp.get_eval_dataset())
    assert sum(len(dataset) for dataset in datasets) == 4
    assert sum(len(dataset.coco.dataset["annotations"]) for dataset in datasets) == 1
    for dataset in datasets:
        assert dataset._classes == ("person",) and dataset.class_ids == [1]
        for index in range(len(dataset)):
            labels = dataset.load_anno(index)
            assert labels.shape[1] == 5
            assert labels[:, 4].tolist() in ([], [0])
    assert sum(len(dataset.load_anno(index)) == 0
               for dataset in datasets for index in range(len(dataset))) == 3


def test_header_validation_does_not_decode_pixels(lan, synthetic, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("header validation must not decode pixels")

    monkeypatch.setattr(cv2, "imread", forbidden)
    monkeypatch.setattr(cvat.Image.Image, "load", forbidden)
    cvat.validate_coco(synthetic, lan.root, workers=3)
    synthetic["images"][2]["height"] += 1
    with pytest.raises(ValueError, match="dimension mismatch"):
        cvat.validate_coco(synthetic, lan.root, workers=3)


@pytest.mark.parametrize("large_app", [False, True])
@pytest.mark.parametrize("full_decode", [False, True])
def test_header_buffer_bounds_underlying_reads(lan, synthetic, monkeypatch, large_app, full_decode):
    path = lan.root / synthetic["images"][0]["file_name"]
    jpeg = path.read_bytes()
    # Legal APP15 segments exceed one small buffer without changing image dimensions.
    app = (b"\xff\xef" + (60002).to_bytes(2, "big") + b"x" * 60000) * 3 if large_app else b""
    payload = jpeg[:2] + app + jpeg[2:] + b"\0" * (2 * 1024 * 1024)
    path.write_bytes(payload)
    original_stat, original_open = Path.stat, Path.open
    read_sizes = []

    def nfs_stat(self, *args, **kwargs):
        info = original_stat(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size, st_blksize=1024 * 1024)
        return info

    class CountedFile(io.BytesIO):
        def readinto(self, buffer):
            size = super().readinto(buffer)
            read_sizes.append(size)
            return size

    def nfs_open(self, mode="r", buffering=-1, *args, **kwargs):
        if self == path and mode == "rb":
            # Emulate CPython's filesystem-dependent default without requiring an NFS mount.
            size = self.stat().st_blksize if buffering == -1 else buffering
            return io.BufferedReader(CountedFile(payload), buffer_size=size)
        return original_open(self, mode, buffering, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", nfs_stat)
    monkeypatch.setattr(Path, "open", nfs_open)
    cvat.validate_coco(synthetic, lan.root, full_decode=full_decode)
    if full_decode:
        assert sum(read_sizes) >= 1024 * 1024  # Opt-in decoding keeps normal buffering.
    else:
        sos = jpeg.index(b"\xff\xda")
        header_size = len(app) + sos + 2 + int.from_bytes(jpeg[sos + 2:sos + 4], "big")
        # Allow the final header read to fill its buffer; do not cap the header itself.
        assert sum(read_sizes) <= ((header_size + 8191) // 8192) * 8192


def test_full_decode_is_opt_in(lan, synthetic):
    path = lan.root / synthetic["images"][0]["file_name"]
    jpeg = path.read_bytes()
    marker = jpeg.index(b"\xff\xda")
    # Preserve the entire header through SOS but remove all entropy-coded pixel data.
    header_end = marker + 2 + int.from_bytes(jpeg[marker + 2:marker + 4], "big")
    path.write_bytes(jpeg[:header_end])
    cvat.validate_coco(synthetic, lan.root)
    with pytest.raises(ValueError, match="truncated"):
        cvat.validate_coco(synthetic, lan.root, full_decode=True)


@pytest.mark.parametrize("credential_source", ["environment", "dotenv"])
@pytest.mark.parametrize("selection", ["project", "tasks"])
def test_supervisor_validates_images_once(lan, synthetic, monkeypatch, run_config,
                                         tmp_path, capsys, credential_source, selection):
    import os
    from tools import train_cvat

    synthetic["annotations"][-1]["category_id"] = 2
    tasks = None
    if selection == "tasks":
        tasks = {}
        for task_id, images in ((101, synthetic["images"][:2]), (103, synthetic["images"][2:])):
            ids = {image["id"] for image in images}
            tasks[task_id] = {**synthetic, "images": images, "annotations": [
                annotation for annotation in synthetic["annotations"] if annotation["image_id"] in ids
            ]}
    install_export(monkeypatch, synthetic, tasks=tasks)
    config_path = tmp_path / "run.toml"
    source = "project_id = 42" if selection == "project" else "task_ids = [101, 103]"
    config_path.write_text(f'[cvat]\nhost = "{HOST}"\n{source}\n'
                           f'[nfs]\nserver = "192.0.2.10"\nframe_root = "{lan.root}"\n'
                           '[dataset]\noutput_dir = "runs"\n[classes]\nexclude = ["person"]\n', encoding="utf-8")
    if credential_source == "dotenv":
        monkeypatch.delenv("CVAT_ACCESS_TOKEN")
    file_token = "test-pat-not-a-real-secret" if credential_source == "dotenv" else "wrong-file-token"
    (tmp_path / ".env").write_text(f'CVAT_ACCESS_TOKEN="{file_token}"\nCVAT_TEST_UNUSED=unused\n', encoding="utf-8")
    monkeypatch.delenv("CVAT_TEST_UNUSED", raising=False)
    monkeypatch.setattr(sys, "argv", ["train_cvat.py", str(config_path)])
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train_cvat.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=lambda *a: 0))
    monkeypatch.setattr(train_cvat, "stop_workers", lambda process: True)
    opened = []
    original = Path.open

    def track(path, *args, **kwargs):
        if path.suffix == ".jpg":
            opened.append(path.relative_to(lan.root).as_posix())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track)

    def launch(command, **kwargs):
        assert "CVAT_ACCESS_TOKEN" not in kwargs["env"]
        assert "CVAT_TEST_UNUSED" not in kwargs["env"]
        with monkeypatch.context() as child:
            child.setenv("YOLOX_CVAT_VALIDATION", kwargs["env"]["YOLOX_CVAT_VALIDATION"])
            cvat.configure_training(get_exp(exp_name="yolox-nano"), command[4])
            assert "YOLOX_CVAT_VALIDATION" not in os.environ
        assert not (Path(command[4]) / "validation.json").exists()
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", launch)
    assert train_cvat.main() == 0
    assert sorted(opened) == sorted(image["file_name"] for image in synthetic["images"])
    captured = capsys.readouterr()
    assert "test-pat-not-a-real-secret" not in str(captured)
    assert "Export contains 4 images" in captured.out
    assert "Preparation complete after" in captured.out
    assert "Run elapsed (including cleanup):" in captured.out
    assert "Downloading: 100%" in captured.err
    assert "Validating images: 100%" in captured.err and "4/4" in captured.err
    summary_path, = Path(run_config["dataset"]["output_dir"]).glob("*/summary.json")
    manifest = json.loads((summary_path.parent / "prepared.json").read_text())
    if selection == "tasks":
        assert manifest["task_ids"] == [101, 103] and "project_id" not in manifest
        assert summary_path.parent.joinpath("resolved_config.json").is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed" and summary["trainer_exit_code"] == 0
    assert summary["source"]["frames"] == 4 and summary["source"]["boxes"] == 4
    assert summary["source"]["boxes_per_class"] == {"car": 3, "person": 1}
    assert summary["dataset"]["retained_boxes"] == 3 and summary["dataset"]["excluded_boxes"] == 1
    assert summary["dataset"]["classes"] == ["car"]
    for split, frames, expected_boxes, expected_empty in [("train", 2, 2, 0), ("val", 1, 1, 0), ("test", 1, 0, 1)]:
        stats = summary["dataset"]["splits"][split]
        assert (stats["frames"], stats["boxes"], stats["empty_frames"]) == (frames, expected_boxes, expected_empty)
        assert stats["boxes_per_class"] == {"car": expected_boxes}
    stats = summary["validation"]
    assert stats["status"] == "complete" and stats["completed_frames"] == 4 and stats["invalid_frames"] == 0
    assert stats["frames_per_second"] == pytest.approx(4 / stats["wall_seconds"])
    assert stats["wall_ms_per_frame"] == pytest.approx(stats["wall_seconds"] * 250)
    latency = stats["worker_latency_ms"]
    assert 0 <= latency["min"] <= latency["median"] <= latency["p95"] == latency["max"]
    assert latency["min"] <= latency["mean"] <= latency["max"]
    assert summary["cleanup"] == {"workers_exited": True, "owned_mount": False, "unmounted": None}
    assert sum(phase["seconds"] for phase in summary["phases"]) + summary["overhead_seconds"] == pytest.approx(summary["total_seconds"])
    assert not any("test-pat-not-a-real-secret" in path.read_text()
                   for path in Path(run_config["dataset"]["output_dir"]).rglob("*.json"))


@pytest.mark.parametrize("change", ["json", "test-json", "mount", "token"])
def test_changed_validation_handoff_cannot_skip_images(lan, synthetic, monkeypatch, tmp_path, change):
    install_export(monkeypatch, synthetic)
    output = tmp_path / "prepared"
    cvat.prepare_dataset(HOST, 42, lan.root, output, nfs_server="192.0.2.10",
                         validation_token="test-handoff")
    if change in {"json", "test-json"}:
        path = output / "annotations" / (cvat.TRAIN_JSON if change == "json" else cvat.TEST_JSON)
        data = json.loads(path.read_text())
        data["images"][0]["width"] += 1
        path.write_text(json.dumps(data))
        message = "dimension mismatch"
    else:
        (lan.root / synthetic["images"][0]["file_name"]).write_bytes(b"")
        message = "non-empty"
        if change == "mount":
            original = cvat.mount_info
            monkeypatch.setattr(cvat, "mount_info", lambda path: {**original(path), "id": 999})
    context = pytest.raises(ValueError, match="No readable frames") if change == "test-json" else nullcontext()
    with context:
        cvat.preflight_dataset(output, validation_token="wrong" if change == "token" else "test-handoff")
    from yolox.data.image_errors import image_errors
    assert any(message in record["reason"] for record in image_errors(output))


def test_disk_cache_estimate_and_space_boundary(tmp_path, monkeypatch):
    splits = {
        "val": [{"file_name": "a/frame.jpg", "width": 100, "height": 300}],
        "train": [{"file_name": "b/frame.jpg", "width": 400, "height": 200}],
        "test": [{"file_name": "c/frame.jpg", "width": 10000, "height": 10000}],
    }
    # Asymmetric sizes catch H/W and split mistakes; held-out test must not reserve training cache.
    expected = 14802  # ceil(((20*60*3+128) + (80*40*3+128)) * 1.1)
    monkeypatch.setattr(cvat.shutil, "disk_usage", lambda path: SimpleNamespace(free=expected))
    assert cvat.check_cache_space(splits, tmp_path / "new", (100, 80), (60, 40)) == expected
    monkeypatch.setattr(cvat.shutil, "disk_usage", lambda path: SimpleNamespace(free=expected - 1))
    with pytest.raises(ValueError, match="Insufficient space.*available.*training.cache"):
        cvat.check_cache_space(splits, tmp_path / "new", (100, 80), (60, 40))


def test_space_failure_precedes_mount_and_training(tmp_path, monkeypatch, run_config, synthetic):
    import os
    from tools import train_cvat

    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(train_cvat.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=lambda *a: 0))
    monkeypatch.setattr(train_cvat, "check_local", lambda path: None)
    monkeypatch.setattr(train_cvat, "check_network_hosts", lambda *args: None)
    monkeypatch.setattr(train_cvat, "export_project", lambda *args: None)
    monkeypatch.setattr(train_cvat, "read_export", lambda *args: synthetic)
    monkeypatch.setattr(cvat.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))

    def forbidden(*args, **kwargs):
        raise AssertionError("disk-space failure must happen before mounting or launching training")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="Insufficient space"):
        train_cvat.run_training(run_config)
    assert not Path(run_config["dataset"]["output_dir"]).exists()


@pytest.mark.parametrize("status,succeeds", [(503, True), (503, False), (401, False)])
def test_export_status_retry_and_credential_redaction(tmp_path, monkeypatch, status, succeeds):
    sdk = pytest.importorskip("cvat_sdk")
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "test-pat-not-a-real-secret")
    destination = tmp_path / "export.zip"
    attempts, delays = [], []

    class Failure(Exception):
        pass

    def client(*args, **kwargs):
        attempts.append(1)
        if succeeds and len(attempts) == 2:
            assert not destination.exists(), "partial export must be removed before retry"
            project = SimpleNamespace(export_dataset=lambda *a, **kw: destination.write_bytes(b"complete"))
            return nullcontext(SimpleNamespace(projects=SimpleNamespace(retrieve=lambda id_: project)))
        destination.write_bytes(b"partial")
        error = Failure("Authorization: test-pat-not-a-real-secret")
        error.status = status
        raise error

    monkeypatch.setattr(sdk, "make_client", client)
    monkeypatch.setattr(cvat.time, "sleep", delays.append)
    with pytest.warns(UserWarning, match="HTTP 503 Service Unavailable") if status == 503 else nullcontext():
        if succeeds:
            cvat.export_project(HOST, 42, destination)
            assert destination.read_bytes() == b"complete"
        else:
            with pytest.raises(ValueError, match=f"HTTP {status}") as error:
                cvat.export_project(HOST, 42, destination)
            import traceback
            assert "test-pat-not-a-real-secret" not in "".join(traceback.format_exception(error.value))
    assert len(attempts) == (2 if status == 503 else 1)
    assert delays == ([3] if status == 503 else [])
