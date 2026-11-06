#!/usr/bin/env bash
# Submit the refit on Unity. Push first (unity/unity_rsync.sh push).
#
# Usage: unity/submit.sh [sbatch options]
# e.g. unity/submit.sh -p cpu,cpu-preempt --requeue

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/unity/config.sh"
ssh "$UNITY_HOST" "cd ${UNITY_WORK}/SelectionSAT && UNITY_WORK=${UNITY_WORK} sbatch -A ${UNITY_ACCOUNT} --export=ALL $* unity/refit.sh"
