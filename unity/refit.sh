#!/bin/bash
#SBATCH -J selectionsat-refit
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -q normal
#SBATCH -o unity/logs/%x-%j.out
#
# Refit the published model: run a copy of selection_bias3.ipynb with the
# to_netcdf line switched on, and keep the executed notebook and the fit.
#
# Submit from the laptop, after unity/unity_rsync.sh push:
#   unity/submit.sh [sbatch options, e.g. -p cpu,cpu-preempt --requeue]
# Outputs, in results/refit-<date>/: selection_bias3_idata.nc,
# selection_bias3_executed.ipynb, figs/.

set -euo pipefail
export TMPDIR="${UNITY_WORK}/tmp"
mkdir -p "$TMPDIR"
source "${UNITY_WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate SelectionSAT

cd "${UNITY_WORK}/SelectionSAT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-} CPUS=${SLURM_CPUS_PER_TASK:-}"
cat unity/UNITY_SYNC_STAMP
python -c "import pymc, arviz, nutpie; print('pymc', pymc.__version__, 'arviz', arviz.__version__, 'nutpie', nutpie.__version__)"

OUT="results/refit-$(date +%Y-%m-%d)"
mkdir -p "$OUT"
cp sat_math_csv_files.zip "$OUT"/    # the notebook skips the download if it's there

# Copy the notebook with the save line switched on
python - "$OUT/selection_bias3.ipynb" <<'PY'
import sys, nbformat
nb = nbformat.read('selection_bias3.ipynb', 4)
n = 0
for c in nb.cells:
    if c.cell_type == 'code' and '# idata3.to_netcdf("selection_bias3_idata.nc")' in c.source:
        c.source = c.source.replace('# idata3.to_netcdf(', 'idata3.to_netcdf(')
        n += 1
assert n == 1, f'expected one to_netcdf line, found {n}'
nbformat.write(nb, sys.argv[1])
print('to_netcdf switched on')
PY

cd "$OUT"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output selection_bias3_executed.ipynb selection_bias3.ipynb
ls -la selection_bias3_idata.nc
echo DONE
