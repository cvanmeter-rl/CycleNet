#!/bin/bash
set -euo pipefail

POD_NAME="cyclenet-pod"
LOCAL_DIR="./figs/real"
REMOTE_DIR="/mnt/project/CycleNet/figs/real"

kubectl exec "$POD_NAME" -- find "$REMOTE_DIR" -type f -print0 | \
while IFS= read -r -d '' remote_path; do
    rel_path="${remote_path#$REMOTE_DIR/}"

    local_path="$LOCAL_DIR/$rel_path"

    mkdir -p "$(dirname "$local_path")"

    echo "Copying $POD_NAME:$remote_path -> $local_path"

    kubectl cp "$POD_NAME:$remote_path" "$local_path"
done
