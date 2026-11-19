"""Prepare CVAT COCO annotations against an existing LAN frame mount."""

import argparse
import sys

from yolox.data.cvat import preflight_dataset, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Export annotations, validate and split by ride")
    prepare.add_argument("--cvat-host", required=True, help="HTTPS CVAT origin (keep TLS hostname)")
    prepare.add_argument("--project-id", required=True, type=int)
    prepare.add_argument("--nfs-server", required=True, help="LAN IPv4 source host")
    prepare.add_argument("--nfs-export", default="/", help="NFSv4 export path")
    prepare.add_argument("--frame-root", required=True, help="Existing read-only NFSv4 assets mount")
    prepare.add_argument("--output", required=True, help="New local dataset directory")
    prepare.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of rides, not frames")
    prepare.add_argument("--seed", type=int, default=0)
    check = subparsers.add_parser("check", help="Rerun preflight without exporting or authenticating")
    check.add_argument("directory")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_dataset(args.cvat_host, args.project_id, args.frame_root,
                            args.output, args.val_fraction, args.seed,
                            nfs_server=args.nfs_server, nfs_export=args.nfs_export)
        else:
            preflight_dataset(args.directory)
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
