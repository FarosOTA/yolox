"""CVAT annotation preparation and Linux LAN/mount preflight (no mount operations)."""

import getpass
import hashlib
from http import HTTPStatus
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import socket
import stat
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
import warnings
import zipfile

from PIL import Image


NFS_IDENTITY = (
    "NFS export maps all clients to nobody (all_squash, anonuid=65534, anongid=65534); "
    "check server-side directory traversal and file read permissions for that identity"
)
LOCAL_FS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "tmpfs", "overlay"}
LAN_NETS = tuple(map(ipaddress.ip_network, (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7",
)))
TAILSCALE_NETS = tuple(map(ipaddress.ip_network, ("100.64.0.0/10", "fd7a:115c:a1e0::/48")))
TRAIN_JSON = "instances_train2017.json"
VAL_JSON = "instances_val2017.json"


def command(*args):
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        raise ValueError(f"Preflight command failed: {args[0]} (install iproute2/util-linux)") from None


def mount_info(path):
    """Resolve symlinks before asking findmnt, including a non-existing output's parent."""
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    mounts = json.loads(command("findmnt", "--json", "--target", str(path),
                                "--output", "TARGET,SOURCE,FSTYPE,OPTIONS,ID"))["filesystems"]
    if len(mounts) != 1:
        raise ValueError(f"Cannot determine mount for {path}")
    return mounts[0]


def check_local(path):
    if mount_info(path)["fstype"] not in LOCAL_FS:
        raise ValueError(f"Prepared data and caches must be on a local filesystem: {path}")


def cvat_hostname(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("CVAT host must be an HTTPS origin without credentials, path or query")
    # Also validates malformed ports before any credential is requested.
    parsed.port
    return parsed.hostname


def check_link(interface):
    try:
        speed = int(Path(f"/sys/class/net/{interface}/speed").read_text().strip())
        duplex = Path(f"/sys/class/net/{interface}/duplex").read_text().strip().lower()
    except (OSError, ValueError):
        warnings.warn(f"Cannot read speed/duplex for {interface}; verify a >=1000 Mb/s full-duplex link")
        return
    print(f"Link {interface}: {speed} Mb/s, {duplex} duplex")
    if speed < 1000 or duplex != "full":
        warnings.warn(f"{interface}: expected >=1000 Mb/s full duplex, got {speed} Mb/s {duplex}")


def check_host(host):
    """Every resolved address must be private LAN, directly routed, and not Tailscale."""
    try:
        addresses = sorted({entry[4][0] for entry in socket.getaddrinfo(
            host, None, type=socket.SOCK_STREAM)})
    except socket.gaierror:
        raise ValueError(f"Cannot resolve LAN host {host}") from None
    if not addresses:
        raise ValueError(f"No addresses for {host}")
    interfaces = set()
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if any(ip in net for net in TAILSCALE_NETS):
            raise ValueError(f"{host} resolves to Tailscale address {address}; use LAN DNS/address")
        if ip.is_global:
            warnings.warn(
                f"{host} resolves publicly to {address}: possible NAT hairpinning. "
                "Use an /etc/hosts entry or local DNS record mapping the hostname to its LAN IP; "
                "keep the CVAT hostname for TLS. NAT hairpinning cannot be proven from routing alone."
            )
        if not any(ip in net for net in LAN_NETS):
            raise ValueError(f"{host}: {address} is not a private LAN address")
        routes = json.loads(command("ip", "-j", "route", "get", address))
        if len(routes) != 1:
            raise ValueError(f"No unambiguous on-link route to {host} ({address})")
        route = routes[0]
        interface = route.get("dev", "")
        if (not interface or interface in {"tailscale0", "lo"} or route.get("gateway")
                or route.get("via") or route.get("type", "unicast") != "unicast"):
            raise ValueError(f"{host} ({address}) is not on-link on the LAN: {route}")
        print(f"LAN {host}: {address} directly via {interface}")
        interfaces.add(interface)
    for interface in sorted(interfaces):
        check_link(interface)


def check_network_hosts(cvat_host, nfs_server):
    """Check LAN routes and the NFS service before any mount operation."""
    check_host(cvat_hostname(cvat_host))
    check_host(nfs_server)
    try:
        with socket.create_connection((nfs_server, 2049), timeout=5):
            pass
    except OSError:
        raise ValueError(f"NFS TCP {nfs_server}:2049 is unreachable on the checked LAN route") from None
    print(f"NFS TCP {nfs_server}:2049 reachable")


def check_network(cvat_host, frame_root, nfs_server, nfs_export="/"):
    check_network_hosts(cvat_host, nfs_server)
    try:
        root = Path(frame_root).resolve(strict=True)
    except PermissionError:
        raise ValueError(f"Cannot access frame root; {NFS_IDENTITY}") from None
    if not root.is_dir():
        raise ValueError(f"Frame root is not a directory: {root}")
    mount = mount_info(root)
    if mount["fstype"] != "nfs4":
        raise ValueError(f"Frame root must be on an existing nfs4 mount, not {mount['fstype']}")
    source = mount["source"]
    if source != f"{nfs_server}:{nfs_export}":
        raise ValueError(f"Expected NFS source {nfs_server}:{nfs_export}, got {source}")
    # The mounted connection may still use an old DNS answer; validate its actual address too.
    options = mount["options"].split(",")
    for option in options:
        if option.startswith("addr="):
            address = option.split("=", 1)[1]
            check_host(address)
            if address != nfs_server:
                raise ValueError(f"Mounted NFS address must be {nfs_server}, got {address}")
    if "ro" not in options:
        raise ValueError("Frame mount must be read-only; mount it with ro yourself before training")
    if "proto=tcp" not in options:
        raise ValueError("Frame mount must use proto=tcp")
    print(f"Frame mount: {mount['fstype']} {source} at {mount['target']}")
    return root


def validate_coco(data, frame_root, *, workers=16, full_decode=False):
    """Stat and inspect headers in parallel; decode pixels only when explicitly requested."""
    images, annotations, categories = (data[key] for key in ("images", "annotations", "categories"))
    ids = [image["id"] for image in images]
    category_ids = [category["id"] for category in categories]
    annotation_ids = [annotation["id"] for annotation in annotations]
    if not images or not categories:
        raise ValueError("COCO must contain images and categories")
    if (len(set(ids)) != len(ids) or len(set(category_ids)) != len(category_ids)
            or len(set(annotation_ids)) != len(annotation_ids)):
        raise ValueError("Duplicate COCO image/category/annotation ids")
    image_ids, class_ids = set(ids), set(category_ids)
    for annotation in annotations:
        if annotation["image_id"] not in image_ids or annotation["category_id"] not in class_ids:
            raise ValueError(f"Annotation {annotation['id']} references an unknown image/category")
    root = Path(frame_root).resolve(strict=True)
    names = set()
    print(f"Validating {len(images)} listed images under {root} "
          f"({'full decode' if full_decode else 'headers only'}, {workers} workers)")
    for image in images:
        name = image["file_name"]
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or len(relative.parts) < 2 or relative.as_posix() != name):
            raise ValueError(f"Expected <ride>/<image> relative to frame root, got {name!r}; "
                             "video-task frame names need an explicit mapping")
        if name in names:
            raise ValueError(f"Duplicate COCO file_name: {name}")
        names.add(name)

    def inspect(image):
        name = image["file_name"]
        path = root / name
        try:
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("image symlink escapes frame root")
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
                raise ValueError("not a non-empty image file")
            # Image.open reads format headers; only the opt-in mode loads pixel data.
            # NFS st_blksize can make default buffering read a MiB for a tiny header.
            with path.open("rb", buffering=-1 if full_decode else 8192) as stream:
                with Image.open(stream) as header:
                    width, height = header.size
                    if full_decode:
                        header.load()
            if (width, height) != (image["width"], image["height"]):
                raise ValueError(f"dimension mismatch: COCO {image['width']}x{image['height']}, "
                                 f"image {width}x{height}")
        except PermissionError:
            return f"{name}: permission denied; {NFS_IDENTITY}"
        except (OSError, ValueError) as error:
            return f"{name}: invalid image header or file: {error}"
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        errors = [error for error in pool.map(inspect, images) if error is not None]
    if errors:
        raise ValueError(f"{len(errors)} invalid image(s):\n" + "\n".join(errors[:20]))


def validation_fingerprint(directory, root):
    """Bind a one-use supervisor handoff to local metadata and this mounted filesystem."""
    files = ["prepared.json", "classes.json", "class_mapping.json",
             f"annotations/{TRAIN_JSON}", f"annotations/{VAL_JSON}"]
    info = root.stat()
    return {
        "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files},
        "root": str(root), "device": info.st_dev, "inode": info.st_ino,
        "mount_namespace": os.stat("/proc/self/ns/mnt").st_ino,
        "mount": mount_info(root),
    }


def validation_handoff(directory, root, token):
    record = directory / "validation.json"
    if not token or not record.exists():
        return False
    try:
        saved = json.loads(record.read_text())
    except (OSError, ValueError):
        return False
    finally:
        record.unlink(missing_ok=True)
    return saved == {"token": token, "fingerprint": validation_fingerprint(directory, root)}


def validation_rides(data, val_fraction, seed):
    rides = sorted({PurePosixPath(image["file_name"]).parts[0] for image in data["images"]})
    if len(rides) < 2:
        raise ValueError("At least two rides are required for a ride-disjoint train/val split")
    random.Random(seed).shuffle(rides)
    count = min(len(rides) - 1, max(1, math.ceil(len(rides) * val_fraction)))
    return set(rides[:count]), set(rides[count:])


def check_cache_space(data, output, input_size, test_size, val_fraction=0.2, seed=0):
    """Budget resized uint8 HWC arrays for both splits, NPY overhead and 10% headroom."""
    val_rides, _ = validation_rides(data, val_fraction, seed)
    sizes = {"train": 0, "val": 0}
    for image in data["images"]:
        split = "val" if PurePosixPath(image["file_name"]).parts[0] in val_rides else "train"
        height, width = image["height"], image["width"]
        target = test_size if split == "val" else input_size
        if height <= 0 or width <= 0 or len(target) != 2 or min(target) <= 0:
            raise ValueError("Image and cache target dimensions must be positive")
        ratio = min(target[0] / height, target[1] / width)
        sizes[split] += int(height * ratio) * int(width * ratio) * 3 + 128
    required = math.ceil(sum(sizes.values()) * 1.1)
    location = Path(output).resolve()
    while not location.exists():
        location = location.parent
    free = shutil.disk_usage(location).free
    detail = (f"disk cache estimate: train {sizes['train'] / 2**30:.2f} GiB, "
              f"val reserve {sizes['val'] / 2**30:.2f} GiB; "
              f"requires {required / 2**30:.2f} GiB including 10% headroom, "
              f"available {free / 2**30:.2f} GiB at {location}")
    print(detail)
    if required > free:
        raise ValueError(f"Insufficient space for {detail}. Free space, choose another local "
                         "dataset.output_dir, reduce input_size/test_size, or use training.cache = 'none'.")
    return required


def read_export(archive):
    """Combine CVAT COCO subsets with independent id spaces without extracting the zip."""
    combined = {"images": [], "annotations": [], "categories": []}
    with zipfile.ZipFile(archive) as bundle:
        files = sorted(name for name in bundle.namelist()
                       if name.startswith("annotations/") and name.endswith(".json"))
        if not files:
            raise ValueError("CVAT export has no annotations/*.json")
        categories = None
        for name in files:
            data = json.loads(bundle.read(name))
            current = sorted(data["categories"], key=lambda category: category["id"])
            if categories is not None and current != categories:
                raise ValueError("CVAT subsets have inconsistent categories")
            categories = current
            id_map = {}
            for image in data["images"]:
                if image["id"] in id_map:
                    raise ValueError(f"Duplicate image id in {name}")
                new_id = len(combined["images"]) + 1
                id_map[image["id"]] = new_id
                combined["images"].append({**image, "id": new_id})
            for annotation in data["annotations"]:
                if annotation["image_id"] not in id_map:
                    raise ValueError(f"Unknown annotation image id in {name}")
                combined["annotations"].append({
                    **annotation, "id": len(combined["annotations"]) + 1,
                    "image_id": id_map[annotation["image_id"]],
                })
        combined["categories"] = categories
    return combined


def export_project(host, project_id, destination):
    # Import lazily: checking a prepared dataset does not need the optional SDK.
    try:
        from cvat_sdk import make_client
        from cvat_sdk.core.proxies.types import Location
    except ImportError:
        raise ValueError("Install the cvat optional extra to export annotations") from None

    token = os.environ.get("CVAT_ACCESS_TOKEN")
    if not token:
        # getpass falls back to echoing input without a TTY; forbid that fallback.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("CVAT personal access token: ")
        except (getpass.GetPassWarning, EOFError):
            raise ValueError("Use an interactive terminal or set CVAT_ACCESS_TOKEN securely") from None
    if not token or not token.strip():
        raise ValueError("A CVAT personal access token is required")
    try:
        for attempt in range(2):
            try:
                with make_client(host, access_token=token) as client:
                    client.projects.retrieve(project_id).export_dataset(
                        "COCO 1.0", destination, include_images=False, location=Location.LOCAL,
                    )
                break
            except Exception as error:
                # Never print SDK bodies/headers/reason strings: even errors may echo a PAT.
                status = getattr(error, "status", None)
                detail = f"HTTP {status} {HTTPStatus(status).phrase}" if status in HTTPStatus else "SDK/transport error"
                if attempt == 0 and status in {429, 500, 502, 503, 504}:
                    warnings.warn(f"CVAT export attempt 1 failed ({detail}); retrying once in 3 seconds")
                    Path(destination).unlink(missing_ok=True)
                    time.sleep(3)
                    continue
                raise ValueError(f"CVAT export failed on attempt {attempt + 1} ({detail}); "
                                 "check TLS, PAT permissions, project id and server. "
                                 "Response bodies/headers omitted to protect credentials.") from None
    finally:
        token = None


def class_merge_sources(groups):
    """Validate target-grouped config and expand it without depending on TOML key order."""
    sources = {}
    for target, labels in groups.items():
        if not target.strip() or not isinstance(labels, list) or not labels:
            raise ValueError(f"classes.merge.{target} must be a non-empty array of source label names")
        for label in labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"classes.merge.{target} contains an invalid source label name")
            if label in sources:
                raise ValueError(f"CVAT label {label!r} appears in both merge groups "
                                 f"{sources[label]!r} and {target!r}")
            sources[label] = target
    return sources


def merge_classes(data, merge):
    """First occurrence in ascending source-id order owns each final class."""
    sources = class_merge_sources(merge)
    categories = sorted(data["categories"], key=lambda category: category["id"])
    names = [category["name"] for category in categories]
    if len(set(names)) != len(names):
        raise ValueError("CVAT category names must be unique for class mapping")
    unknown = set(sources) - set(names)
    if unknown:
        raise ValueError(f"Unknown CVAT labels in classes.merge: {', '.join(sorted(unknown))}")
    for target, labels in merge.items():
        if target in names and target not in labels:
            raise ValueError(f"Merge target {target!r} is an existing CVAT label; "
                             "include it explicitly in its own group to merge it")
    classes, remap, mapping = {}, {}, []
    for category in categories:
        name = sources.get(category["name"], category["name"])
        if name not in classes:
            classes[name] = len(classes) + 1
        target_id = classes[name] if merge else category["id"]
        remap[category["id"]] = target_id
        mapping.append({"source_id": category["id"], "source": category["name"],
                        "class": name, "category_id": target_id, "class_index": classes[name] - 1})
    if merge:
        data["categories"] = [{"id": id_, "name": name} for name, id_ in classes.items()]
        for annotation in data["annotations"]:
            annotation["category_id"] = remap[annotation["category_id"]]
    else:
        data["categories"] = categories
    print("Final source-to-class mapping: " + json.dumps(mapping, ensure_ascii=False))
    return mapping


def prepare_dataset(cvat_host, project_id, frame_root, output, val_fraction=0.2, seed=0,
                    *, nfs_server, merge=None, nfs_export="/", data=None,
                    workers=16, full_decode=False, validation_token=None):
    cvat_hostname(cvat_host)
    if not 0 < val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output must not already exist; use a new local directory")
    check_local(output)
    root = check_network(cvat_host, frame_root, nfs_server, nfs_export)
    if data is None:
        with tempfile.TemporaryDirectory(prefix="yolox-cvat-") as temporary:
            archive = Path(temporary) / "annotations.zip"
            export_project(cvat_host, project_id, archive)
            data = read_export(archive)
    validate_coco(data, root, workers=workers, full_decode=full_decode)
    mapping = merge_classes(data, merge or {})
    val_rides, train_rides = validation_rides(data, val_fraction, seed)
    output.mkdir(parents=True)
    (output / "annotations").mkdir()
    for split, filename in (("train", TRAIN_JSON), ("val", VAL_JSON)):
        images = [image for image in data["images"]
                  if (PurePosixPath(image["file_name"]).parts[0] in val_rides) == (split == "val")]
        ids = {image["id"] for image in images}
        subset = {**data, "images": images, "annotations": [
            annotation for annotation in data["annotations"] if annotation["image_id"] in ids
        ]}
        (output / "annotations" / filename).write_text(json.dumps(subset), encoding="utf-8")
        (output / f"{split}2017").symlink_to(root, target_is_directory=True)
    classes = [category["name"] for category in data["categories"]]
    (output / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    (output / "class_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    manifest = {
        "cvat_host": cvat_host, "project_id": project_id, "frame_root": str(root),
        "nfs_server": nfs_server, "nfs_export": nfs_export,
        "num_classes": len(classes), "seed": seed, "val_fraction": val_fraction,
        "train_rides": sorted(train_rides), "val_rides": sorted(val_rides),
        "validation_workers": workers, "full_decode": full_decode,
    }
    (output / "prepared.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if validation_token:
        (output / "validation.json").write_text(json.dumps({
            "token": validation_token, "fingerprint": validation_fingerprint(output, root),
        }), encoding="utf-8")
    print(f"Prepared {len(data['images'])} images; num_classes={len(classes)}; classes={classes}")
    print(f"Train rides: {manifest['train_rides']}; validation rides: {manifest['val_rides']}")
    return manifest


def preflight_dataset(directory, *, validation_token=None):
    """Recheck images unless this launcher consumes the supervisor's matching one-use record."""
    directory = Path(directory).resolve(strict=True)
    check_local(directory)
    manifest = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    root = check_network(manifest["cvat_host"], manifest["frame_root"],
                         manifest["nfs_server"], manifest["nfs_export"])
    validated = validation_handoff(directory, root, validation_token)
    if validated:
        print("Reusing supervisor image validation (matching mount and JSON hashes; one-use handoff)")
    subsets = []
    for split, filename in (("train", TRAIN_JSON), ("val", VAL_JSON)):
        if (directory / f"{split}2017").resolve(strict=True) != root:
            raise ValueError(f"{split}2017 must point to the configured frame root")
        cache = directory / f"cache_{split}2017"
        if cache.is_symlink() or (cache.exists() and any(p.is_symlink() for p in cache.rglob("*"))):
            raise ValueError(f"Cache must not contain symlinks: {cache}")
        check_local(cache)
        data = json.loads((directory / "annotations" / filename).read_text(encoding="utf-8"))
        if not validated:
            validate_coco(data, root, workers=manifest.get("validation_workers", 16),
                          full_decode=manifest.get("full_decode", False))
        subsets.append(data)
    classes = [c["name"] for c in sorted(subsets[0]["categories"], key=lambda c: c["id"])]
    if (subsets[0]["categories"] != subsets[1]["categories"]
            or len(classes) != manifest["num_classes"]
            or json.loads((directory / "classes.json").read_text(encoding="utf-8")) != classes):
        raise ValueError("Prepared class metadata is inconsistent")
    rides = [{PurePosixPath(image["file_name"]).parts[0] for image in subset["images"]}
             for subset in subsets]
    if rides[0] & rides[1]:
        raise ValueError("Train/val rides overlap")
    print(f"Preflight passed: num_classes={len(classes)}, "
          f"train={len(subsets[0]['images'])}, val={len(subsets[1]['images'])}")
    return manifest


def configure_training(exp, directory, *, cache=None):
    manifest = preflight_dataset(directory, validation_token=os.environ.pop("YOLOX_CVAT_VALIDATION", None))
    if cache == "disk":
        images = []
        for filename in (TRAIN_JSON, VAL_JSON):
            images.extend(json.loads((Path(directory) / "annotations" / filename).read_text())["images"])
        check_cache_space({"images": images}, directory, exp.input_size, exp.test_size,
                          manifest["val_fraction"], manifest["seed"])
    exp.data_dir = str(Path(directory).resolve())
    exp.train_ann = TRAIN_JSON
    exp.val_ann = VAL_JSON
    exp.num_classes = manifest["num_classes"]
