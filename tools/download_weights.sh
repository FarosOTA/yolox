#!/usr/bin/env bash
# Download official COCO-pretrained YOLOX checkpoints, excluding legacy Darknet53.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
weights_dir="$repo_root/weights"
base="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
mkdir -p "$weights_dir"

temp_file=""
trap 'if [[ -n "$temp_file" ]]; then rm -f -- "$temp_file"; fi' EXIT

for model in nano tiny s m l x; do
    file="$weights_dir/yolox_${model}.pth"
    if [[ -s "$file" ]]; then
        printf 'Skipping existing checkpoint: %s\n' "$file"
        continue
    fi

    printf 'Downloading YOLOX-%s to %s\n' "$model" "$file"
    temp_file="$(mktemp "$file.part.XXXXXX")"
    curl --fail --location --retry 3 \
        "$base/yolox_${model}.pth" --output "$temp_file"
    mv -- "$temp_file" "$file"
    temp_file=""
done
