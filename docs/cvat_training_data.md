# CVAT annotations and LAN training frames

This fork adds a Linux workflow that exports **COCO 1.0 annotations without
images** using the optional `cvat-sdk` extra, reads only the exported filenames
over read-only NFSv4, splits images 70% train / 15% validation / 15% held-out test,
and supervises normal-user training.
It can own a temporary mount or reuse an existing mount. It never configures
the server, copies the frame tree locally, or modifies source images.

## Install and configure

The training client needs Linux, `ip` (iproute2), `findmnt` (util-linux),
readable `/sys/class/net/<interface>/{speed,duplex}`, and NFS mount helpers
(for example `nfs-common`). CUDA is the recommended training profile for this
workflow. The general training CLI also supports single-device FP32 CPU/MPS,
but this Linux supervisor is not ported to macOS. Install with:

```sh
uv sync --locked --extra cuda --extra export --extra cvat
```

Use `--extra cpu` **instead of** `--extra cuda` for preparation and tests without
a compatible NVIDIA driver. The SDK is constrained to CVAT 2.74 and locked to
2.74.1. Preflight-only checks need neither the SDK nor a PAT.

Copy the [minimal example](../train.toml.example) to a private root-level config:

```sh
cp train.toml.example train.toml
```

Edit the placeholders before running. `train.toml` and
`configs/cvat/*.local.toml` are gitignored;
a config outside the repository works too. **Never put a PAT in the config,
even in comments.** The example uses reserved documentation addresses, not
routable deployment defaults. The workflow has **no default CVAT host, project
ID or NFS server**. Keep real configs, run records and annotations out of version
control; the default example output lives under ignored `datasets/`.

The only supervisor argument is the TOML config path:

```sh
sudo -v
uv run --no-sync python tools/train_cvat.py train.toml
```

Run as your **normal user**, never with sudo around Python. `sudo -v` is a
separate interactive authentication step for temporary mounting; omit it for
an already-mounted share. The workflow uses `sudo -n` and will **not prompt
for a sudo password** in interactive or non-interactive sessions.

### Config tables

All filesystem paths resolve relative to the config's parent directory, not
the shell's working directory. Unknown tables, keys and experiment override
names fail before network or credential operations, so typos cannot silently
use defaults.

The example keeps only the usual per-run choices visible. Output location,
batch size, NFS export/options, validation tuning, GPU count, precision and cache mode are optional advanced
settings; omit them to use the defaults below. A root-level config uses paths
such as `datasets/cvat-runs` and `weights/yolox_nano.pth`, without `../../`.

| Table | Settings |
| --- | --- |
| `cvat` | Required HTTPS `host` and non-empty `task_ids` array of unique positive integers. The template's IDs are fixtures; replace them with your tasks. Only selected tasks are exported and combined, then split 70/15/15 by image count. Legacy positive integer `project_id` is an alternative, never combined with `task_ids`. Keep the hostname for TLS; no credentials, path or query in the URL. |
| `nfs` | Required LAN IPv4 `server`; `export` defaults to `/`; `options` defaults to `["ro", "vers=4.2", "proto=tcp"]`. These three options are mandatory; only `nosuid`, `nodev`, `noexec`, `hard` may be added. Optional `frame_root` reuses an existing mount and never mounts/unmounts it. |
| `dataset` | Local `output_dir` base (default `datasets/cvat-runs`), integer split `seed` (default 0), `validation_workers` (16, allowed 1–128), `full_decode` (false). The image split is fixed at 70/15/15; the old `val_fraction` setting is rejected. Each invocation creates a unique UTC timestamp/UUID run folder. |
| `training` | `model` (default `yolox-nano`) **or** `exp_file`, `devices` (1), `batch` (8), `fp16` (false), `cache` (`disk`, `ram` or `none`), optional `checkpoint` for fine-tuning. There is no `resume` setting. |
| `training.overrides` | Experiment settings such as `max_epoch = 100`, `input_size = [640, 640]` and `data_num_workers = 4`. The example omits overrides, keeping the selected experiment's defaults (300 epochs for Nano). Dataset paths, class count and output directory are managed by the workflow, not overridable here. |
| `classes.exclude` | Optional array of original CVAT labels to remove (default empty). Images remain, but their excluded boxes become unlabelled/background. |
| `classes.merge` | Optional target-class groups as described below. Exclusions cannot overlap group sources or targets. |

Model checkpoints/logs go under `<run>/training`. Disk caches stay in the
local run folder, not on NFS. Every supervisor invocation starts a new run:
`training.checkpoint` initializes compatible model weights, not the optimizer,
epoch or training state. Omit it to train from scratch; weights are not
downloaded automatically. Old configs containing `training.resume` are rejected
rather than silently changing their meaning. For continuation, use the lower-level
training CLI's `--resume` with persistent prepared data (see below).
Use experiments honoring `data_dir`, `train_ann`, `val_ann` and `num_classes`.

### Class merging and deterministic indices

Exclusions and any number of target groups can be combined:

```toml
[classes]
exclude = ["cone"]

[classes.merge]
vehicle = ["car", "truck", "bus"]
two_wheeler = ["bicycle", "motorcycle"]
```

Exclusions refer to **original CVAT labels**, not post-merge names. Their
categories and boxes are removed before splitting. All images remain, including
images with no retained boxes. Excluded objects become unlabelled/background;
this is **not** an ignore-region mechanism. No source images or CVAT annotations
are modified on the server. Omit `classes.exclude` to preserve all classes.

Unknown or duplicate exclusions, empty/non-string label names, overlap with any
merge source or target, and exclusions leaving no classes fail explicitly.
Unlisted labels retain their names. Unknown source labels, empty groups and
sources appearing in multiple groups fail with explicit errors. An existing
CVAT label cannot be a target unless explicitly listed in its own group:
if `vehicle` already exists, use `vehicle = ["vehicle", "car", "truck", "bus"]`.
This prevents accidental merging of an untouched class into a group.

Class order follows the **first occurrence of each final class while walking
original categories in ascending CVAT category-id order**. It does not depend
on TOML group order or source-array order. For source ids `car=1`, `person=3`,
`truck=4`, `bicycle=7`, `motorcycle=9`, `cone=11`, `bus=12`, the above groups
produce `["vehicle", "person", "two_wheeler"]` at indices 0–2 with `cone`
excluded. An untouched label's index can shift when preceding labels merge
or are excluded.

With merging or exclusions enabled, surviving categories are renumbered 1–N
and every retained annotation's `category_id` is rewritten in **all three** train,
validation and test JSON files. Without either, sorted original category ids are retained.
`classes.json`, reported
`num_classes`, loaders and per-class AP all use the final names/indices.
Every box of retained labels is preserved. The final source-to-class mapping
is printed and saved; excluded source labels have `null` for `class`,
`category_id` and `class_index`. `resolved_config.json` also records the exact
exclusion list and merge groups.

## Credentials and temporary export

For the TOML supervisor, store `CVAT_ACCESS_TOKEN` in `.env` beside the TOML
file (not relative to the current working directory):

```dotenv
CVAT_ACCESS_TOKEN=your-personal-access-token
```

Edit the file locally rather than placing your token in shell commands or chat.
The repository ignores `.env`; restrict its permissions with `chmod 600 .env`.
For configs outside this repository, ensure their `.env` is also excluded from
version control. Only `CVAT_ACCESS_TOKEN` is loaded; other keys are not imported,
and `${...}` interpolation is disabled so tokens are read literally.
An existing non-empty exported token takes precedence over `.env`. Missing or
empty tokens fall back to the hidden prompt. The standalone preparation tool
uses the environment or prompt, not automatic `.env` loading.

Alternatively, for a Bash session with tracing disabled:

```sh
set +x
read -r -s -p 'CVAT PAT: ' CVAT_ACCESS_TOKEN; printf '\n'
export CVAT_ACCESS_TOKEN
# Run the workflow, then:
unset CVAT_ACCESS_TOKEN
```

There is no token CLI argument or config field. Never put the PAT in URLs,
scripts, tracked files, logs or shell history. The supervisor omits the token
from run records and the training subprocess environment.
A terminal without secure no-echo input
fails rather than falling back to echoed input. SDK failures show the HTTP
status and its standard description when available; response bodies, headers
and raw exception strings remain suppressed because they may echo credentials.
HTTP 429/500/502/503/504 failures retry once after three seconds, removing a
partial archive first. The first failure warns and repeated failure is an
error; authentication failures and opaque transport errors do not retry.
This does not diagnose the cause of a transient server failure. TLS verification stays
enabled; install a trusted CA if needed, rather than using a bare IP or
disabling verification. The SDK exports the project as COCO 1.0 without images
into a temporary directory removed on success or failure.

## Network checks and mount lifecycle

Before mounting, both the CVAT hostname and NFS server must resolve entirely
to private LAN addresses and have on-link routes without gateways. Every
IPv4/IPv6 answer is checked; a good answer cannot mask a bad one. Tailscale
ranges (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`), `tailscale0`, loopback/local
routes, public answers and gateway routes fail. Merely having Tailscale
enabled does not fail if actual traffic stays on the LAN.

A public DNS answer emits a **possible NAT hairpinning** warning before
failing. Routing alone cannot prove NAT. Use local DNS or an `/etc/hosts`
entry to map the HTTPS hostname to the LAN server, for example
`192.0.2.10 cvat.example.com` (replace both documentation placeholders).

Speed and duplex are read on each interface selected by `ip -j route get`.
Below 1000 Mb/s, half/unknown duplex or unavailable metadata warns rather
than fails. The client cannot measure the server NIC, switch uplinks or
end-to-end throughput. A TCP connection to the checked NFS server on port
2049 must succeed (five-second timeout) **before mounting**. This is a
reachability check, not an NFS RPC handshake. NFSv4-only servers do not need
rpcbind/111 or network mountd; `showmount` is not a precondition.

Temporary mode creates a private directory and runs only the mount/unmount
operations through `sudo -n`. An administrator must authorize those
operations beforehand for unattended use. No sudoers configuration is
installed and no wildcard rule for user-controlled temporary paths is
recommended. Cached sudo authorization may expire during long training;
cleanup then retains the mount and prints manual recovery instructions.
Training and DataLoader workers never run as root.

After mounting, preflight requires `findmnt` to report `nfs4`, the configured
`server:export`, `ro` and `proto=tcp`. The mounted `addr=` is also checked
on-link and must match the configured server. Image checks then run before
training. The launcher rechecks network/mount and local metadata before caching
and worker launch, but consumes the supervisor's one-use validation record to
avoid a second image pass. The record binds SHA-256 hashes of both annotation
JSONs and class/provenance metadata to the frame root, filesystem device/inode,
mount ID/options/source and mount namespace. It is not a reusable validation
cache: a missing/changed record, JSON or mount triggers image checks again.
Standalone `--prepared-data` invocations without the supervisor handoff always
check images. Runtime COCO reads also detect decode failures and dimension changes,
but not arbitrary content changes; keep the share immutable throughout a run.

On normal completion, exceptions, failed preflight, Ctrl-C or SIGTERM, the
supervisor terminates the training process group and waits/reaps DataLoader
descendants, including orphans via Linux subreaper support, **before unmount**.
Shutdown allows ten seconds for SIGTERM then ten seconds for SIGKILL.
Signals during mount/spawn are deferred until the helper/child is tracked;
additional interrupts during cleanup are ignored. It never lazy/force
unmounts underneath workers. Stuck workers or failed unmount leave the
mountpoint intact and report the exact source, target and recovery commands:
`sudo umount -- <mountpoint>` then `rmdir -- <mountpoint>` after workers exit.
SIGKILL of the supervisor, power loss or kernel failure cannot run cleanup;
inspect `findmnt` for `yolox-nfs-*` mountpoints after such an interruption.

Run records remain after shutdown, but temporary-mode image symlinks and the
recorded frame root refer to the removed mount. Start a new supervised run
rather than reusing those links. For persistent prepared data, set
`nfs.frame_root` and manage that mount separately.

## Existing mount and optional NFS setup

For an assets-only export with `fsid=0`, mount `server:/`, **not** its
server-side path. Example manual client setup (replace the documentation IP):

```sh
sudo mkdir -p /mnt/training-frames
sudo mount -t nfs -o ro,vers=4.2,proto=tcp 192.0.2.10:/ /mnt/training-frames
findmnt -T /mnt/training-frames -o TARGET,SOURCE,FSTYPE,OPTIONS
```

Set `frame_root = "/mnt/training-frames"` under `[nfs]`. The workflow never
unmounts a user-supplied root. Optional administrator-managed `/etc/fstab`:

```text
192.0.2.10:/ /mnt/training-frames nfs ro,vers=4.2,proto=tcp,_netdev,nofail,x-systemd.automount 0 0
```

Systemd may mount this on first access; that is configured system behavior,
not the workflow mounting an existing root. For a manual-only mount omit
the fstab entry. A server administrator can optionally export only the
assets directory read-only to the training client, for example:

```text
/srv/training-assets 192.0.2.15(ro,sync,no_subtree_check,all_squash,anonuid=65534,anongid=65534,fsid=0)
```

Configure NFSv4/TCP-only and firewall access to trusted clients separately;
this tool does not change exports/services/firewalls and has no CIFS path.
AUTH_SYS is neither encryption nor strong client authentication.
`all_squash` maps reads to nobody (UID/GID 65534). Traversal/read permissions
must allow that identity; permission errors name this mapping. Running
training as root does not bypass it.

## Dataset checks and run records

Filenames must be ride-relative, for example
`20250101_120000/20250101_120000_0123abcd.jpg`, resolving directly under the
configured root. No extra `assets` prefix is inserted or stripped. Bare
video-task `frame_000000.jpg` names require external mapping and fail.
Only exported images are read: no recursive frame scan, sibling folders or
unlisted images enter training. Every listed path must be a readable, regular,
non-empty file confined to the frame root. Stats and Pillow header reads run
in parallel (16 workers by default); dimensions must match JSON. Validation checks
report completed images, percentage, elapsed time, throughput and estimated time
remaining, plus an error counter when errors are found. Updates are throttled
to about once a second while work completes, with an immediate initial and final
display. Completed checks count even when an earlier image is slow; the final
error list remains in export order. Progress bars go to stderr and remain enabled
for redirected output. Phase messages on stdout are flushed immediately; the
supervisor reports preparation and total elapsed time including cleanup.
CVAT export announces the server-preparation wait (no percentage/ETA is available
for that wait), then reports download progress and total export time.

The supervisor prints a summary on completion, failure or handled interruption.
It includes model/checkpoint/cache settings and resolved epochs, dimensions and
worker counts; source frame/box/class counts; retained/excluded boxes; and train/val/test
frames, empty frames, rides and per-class box counts. Phase durations and percentages
are **exclusive wall times**, not overlapping parent/child timers: local/LAN
preflight, export/download, annotation parsing, experiment/cache capacity, optional
mount, mounted-source preflight, image validation, class mapping/splitting, dataset
and config writing, trainer subprocess, and cleanup. Other setup/reporting overhead
is shown separately so the breakdown accounts for total elapsed time. Measurement
ends after cleanup, before rendering/saving the summary. Trainer subprocess time
includes startup, caching, epochs and evaluation; it is **not** a separate measurement
of GPU compute or individual training phases. Existing trainer logs retain iteration
timing, ETA and evaluation metrics.

Frame statistics describe **validation**, not training/inference performance:
completed/total checks, valid/invalid counts, mode and workers, frames/s, wall-clock
ms per completed frame, and per-worker check latency min/mean/median/p95/max in ms.
Worker latency excludes queue wait; wall averages include parallelism and validation
overhead. P95 uses the nearest-rank definition. On interruption, counts and latency
samples include only completions observed by the progress loop; checks finishing
during executor shutdown are not added. Partial or unreached phases are not reported
as completed work. Cleanup failures make the overall result failed, even if the
trainer exited successfully. Errors identify the phase/type without recording raw
exception text or credentials. Hard kills/process crashes cannot guarantee a summary.

Header reads use an explicit 8 KiB Python buffer rather than the filesystem's default
(which can be much larger on NFS). This is a buffer size, not a header-size
limit: JPEGs with large APP/EXIF segments can read additional buffers. It does
not control kernel/NFS readahead. Opt-in full decoding keeps normal buffering.
No pixels are decoded in the default mode. Header checks cannot detect corruption
in pixel payloads; set `dataset.full_decode = true` for slower full decoding,
still in parallel and once per supervised run. Preparation skips missing, empty,
unreadable or invalid images (including dimension mismatches) and removes their
annotations before splitting. Unsafe paths, escaping symlinks and malformed COCO
metadata remain fatal. Too few usable images to populate all splits also fail.

COCO cache construction and loading tolerate frames that become unreadable after
validation, including OpenCV returning no pixels. Cache estimation/building skips
only source-image read failures; cache write failures and unrelated errors still
propagate. Missing entries in an interrupted disk cache fall back to source reads.
At runtime, the loader skips a failed frame and uses the next readable frame **with
its own labels, dimensions and image ID**, keeping batch sizes stable for mosaic
and distributed training. It stops with an error if no readable frames remain.
Model evaluation deduplicates replacement IDs and evaluates only successfully read
images, including readable images with zero detections. Prepared JSON and split
counts describe preparation, not the runtime replacement frequency.

Verbose warnings at the end of image validation, model validation and training list
every recorded path, image ID, split and failure reason without a 20-entry truncation.
Atomic, deduplicated JSON records in the local dataset's `skipped_frames/` directory
survive loader processes and trainer failures; the supervisor includes them in
`summary.json` and repeats them after worker/mount cleanup. Records accumulate for
the prepared directory, including when reusing it for another invocation; they are
diagnostics, not a permanent blacklist of repaired files. A loader instance remembers
failed source frames until it is recreated. Repair the listed files or investigate
NFS health; skipping a frame does not repair its source.

COCO subsets are combined with image/annotation ids remapped. Duplicate
filenames/ids, inconsistent categories and unknown references fail. Every new
dataset uses **70% train / 15% validation / 15% test by image count**, regardless
of task count or sizes. Whole-image counts use largest-remainder rounding:
floor each target, then assign leftover images to the largest fractional parts
(ties use train, validation, test order). Every bucket receives at least one
image; for tiny datasets this takes images from training if needed.
At least three images are required, even if
there is only one task/ride. For 2,539 images the counts are 1,777 / 381 / 381.

Images are sorted by ride-relative filename. Each ride is apportioned against
the remaining global split quotas using largest remainders, then divided into
at most three contiguous filename segments. `dataset.seed` determines their
train/val/test order. Export ordering does not affect membership. Timestamp-prefixed
filenames preserve temporal order; arbitrary filenames may not. No images are
dropped or shared between buckets, including unannotated images. This is **not
ride-disjoint or class-stratified**: segments from a ride may appear in every
bucket, and similar frames can cross segment boundaries. It reduces mixing
compared with random per-frame shuffling but does not guarantee independence.
Training updates weights from train and selects checkpoints using validation;
test is never used in the training loop or its cache budget. Keep test evaluation
for the final selected checkpoint, not iterative tuning.

Each unique run folder contains:

- `annotations/instances_train2017.json`, `instances_val2017.json` and `instances_test2017.json`;
- `train2017`, `val2017` and `test2017` symlinks to the same frame root, with disjoint image selections;
- `classes.json` in model index order and `class_mapping.json` with each source id/name, target name/id/index;
- `prepared.json` with server/task IDs (or legacy project ID), NFS, split ratios/method/counts/seed, rides and class metadata;
- `resolved_config.json` with validated defaults, resolved paths, grouped merges and generated training arguments (no PAT or raw config comments);
- `summary.json` after supervised cleanup, with the run outcome, phase durations, frame statistics, dataset counts and cleanup result; failures before a run directory is created print only a terminal summary;
- transient `validation.json` for the one-use supervisor-to-launcher handoff (a random handoff identifier, **not** a CVAT PAT; consumed by the launcher);
- local `cache_train2017` when using disk cache and `training/` for model outputs.

Records are private operational data even without tokens. Do not commit them.
Caches/output must be local; network filesystems and cache symlinks fail.
Existing YOLOX caches are not content-addressed: do not mutate data during a
run or reuse stale caches after image changes. Preflight is a point-in-time
check, not a snapshot.

For disk caching, the supervisor exports annotations **before mounting** and
estimates resized uint8 HWC array bytes using each listed image's dimensions,
the train/validation image selections and resolved experiment `input_size`/`test_size`. It includes
128 bytes per NPY header and 10% headroom, compares with free space on the
local output filesystem, and fails before mounting/training if insufficient.
The standard experiment currently caches only training images: the reported
validation amount is a conservative reserve, not an extra cache it creates.
The launcher rechecks capacity before caching, including standalone
`--prepared-data --cache disk` use. The estimate does not reserve disk space
or budget checkpoints/logs indefinitely; free space can change concurrently.
Free space, select a larger local output filesystem, lower the configured
resolutions, or set `training.cache = "none"` when capacity is insufficient.

## Standalone preparation and rechecks

The lower-level existing-mount tool requires a new local output directory;
the supervisor above creates it automatically. This tool does not select or merge
classes; use the TOML supervisor for per-run exclusions and merging.

```sh
uv run --no-sync python tools/prepare_cvat.py prepare \
  --cvat-host https://cvat.example.com --project-id 42 \
  --nfs-server 192.0.2.10 --nfs-export / \
  --frame-root /mnt/training-frames --output datasets/prepared-cvat
uv run --no-sync python tools/prepare_cvat.py check datasets/prepared-cvat
uv run --no-sync python tools/train.py -n yolox-nano -d 1 -b 8 \
  --prepared-data datasets/prepared-cvat --cache disk
```

Replace documentation placeholders. `--prepared-data` configures training/validation
and class count after experiment overrides, and reruns preflight on all three splits before
cache/CUDA/distributed launch. Use it on resume and every training node.
Without it, existing training behavior is unchanged. Checkpoints are not
downloaded automatically.

After selecting the final checkpoint, evaluate the labeled held-out test set
with the same model/experiment and input-size overrides used for training:

```sh
uv run --no-sync python tools/eval.py -n yolox-nano -d 1 -b 8 \
  --prepared-data datasets/prepared-cvat --split test \
  -c path/to/best_ckpt.pth
```

The default `--split val` uses validation. Do not combine prepared data with
`--test`: that upstream flag means unlabeled COCO test-dev, not this labeled
test set. The CVAT path retains normal COCO AP/AR computation. It requires a
readable existing NFS mount; a supervisor-owned temporary mount has already
been unmounted at run completion. Reuse/mount the recorded frame root before
standalone evaluation. Existing two-way prepared datasets remain readable,
but must be prepared again to obtain the new 70/15/15 split and test set.

To resume interrupted training on persistent prepared data, keep the same
experiment settings, classes, source data and split, and pass the checkpoint
explicitly. For example:

```sh
uv run --no-sync python tools/train.py -n yolox-nano -d 1 -b 8 \
  --prepared-data datasets/prepared-cvat --cache disk \
  --resume -c path/to/latest_ckpt.pth
```

This is distinct from initializing a fresh supervised run with COCO weights.
Temporary mounts removed by the supervisor are not persistent prepared data;
use an existing mount/prepared directory whose image links remain valid.

## Validation scope

Tests use fake networking/mounts and a synthetic multi-subset COCO export,
real decoded images, YOLOX train/validation loaders, local disk caching and
COCO evaluation. They cover gateway/Tailscale/hairpin rejection, link warnings,
image failures, permissions, config typos/paths, exclusions (including empty
images), grouped merging/order/AP
names, unique runs/records, and lifecycle cleanup including signals, partial
mount failure, stuck workers and failed unmount. A repository privacy test
scans tracked and non-ignored new files. SDK tests need the `cvat` extra;
Linux workflow tests skip on other platforms.

```sh
uv sync --locked --extra cpu --extra export --extra cvat
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv lock --check
git diff --check
```

Tests do not establish a real NFS
mount, CVAT connection, LAN throughput or a CUDA training run. A live client
run must still validate its own mount and exported images before training.
