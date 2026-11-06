#!/usr/bin/env bash
# Push the refit's inputs to Unity and pull the results back.
#
# Unity is compute only: no git there. Push sends the published notebook, the
# parsed data, environment.yml and these scripts; pull brings back results/
# (the fitted InferenceData and the executed notebook) and the logs.
#
# Usage (from anywhere):
#   unity/unity_rsync.sh push [--dry-run]
#   unity/unity_rsync.sh pull [--dry-run]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# UNITY_HOST and UNITY_WORK (see unity/config.example.sh)
source "${ROOT}/unity/config.sh"
REMOTE_DIR="${UNITY_WORK}/SelectionSAT"
REMOTE="${UNITY_HOST}:${REMOTE_DIR}"
STAMP="${ROOT}/unity/UNITY_SYNC_STAMP"

cmd="${1:-}"
shift || true
DRY=()
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY=(-n) ;;
  esac
done

case "$cmd" in
  push)
    dirty=$(git -C "$ROOT" status --porcelain -- selection_bias3.ipynb sat_math_csv_files.zip environment.yml unity | grep -c . || true)
    {
      echo "pushed_at=$(date -Is)"
      echo "git_head=$(git -C "$ROOT" rev-parse HEAD)"
      echo "dirty_files=${dirty}"
      echo "notebook_md5=$(md5sum "$ROOT/selection_bias3.ipynb" | cut -d' ' -f1)"
    } >"$STAMP"
    ssh "$UNITY_HOST" "mkdir -p ${REMOTE_DIR}/unity/logs ${REMOTE_DIR}/results"
    rsync -avP "${DRY[@]}" "$ROOT"/selection_bias3.ipynb "$ROOT"/sat_math_csv_files.zip "$ROOT"/environment.yml "$REMOTE"/
    rsync -avP "${DRY[@]}" --delete --exclude logs/ --exclude config.sh "$ROOT"/unity/ "$REMOTE"/unity/
    ;;
  pull)
    mkdir -p "$ROOT"/results "$ROOT"/unity/logs
    rsync -avP "${DRY[@]}" "$REMOTE"/results/ "$ROOT"/results/
    rsync -avP "${DRY[@]}" "$REMOTE"/unity/logs/ "$ROOT"/unity/logs/
    ;;
  *)
    echo "Usage: $0 {push|pull} [--dry-run]" >&2
    exit 2
    ;;
esac
