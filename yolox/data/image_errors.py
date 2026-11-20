"""Durable frame-read diagnostics shared by cache threads and loader processes."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from loguru import logger


class ImageReadError(OSError):
    """A source frame could not be read or decoded (not a cache write failure)."""


def record_image_error(directory, record):
    directory = Path(directory) / "skipped_frames"
    directory.mkdir(exist_ok=True)
    payload = json.dumps(record, sort_keys=True)
    destination = directory / (hashlib.sha256(payload.encode()).hexdigest() + ".json")
    # Publish complete records atomically, including while workers are prefetching.
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload + "\n")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def image_errors(directory):
    return [json.loads(path.read_text())
            for path in sorted((Path(directory) / "skipped_frames").glob("*.json"))]


def warn_image_errors(records, context):
    if records:
        logger.warning(
            "{}: skipped unreadable frames ({} distinct diagnostics). "
            "Training/metrics exclude these reads; inspect the source files and NFS health.\n{}",
            context, len(records),
            "\n".join(
                f"  split={r['split']} image_id={r['image_id']} path={r['path']} reason={r['reason']}"
                for r in records
            ),
        )
