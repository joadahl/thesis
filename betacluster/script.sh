#!/usr/bin/env bash

SOURCE_PATH="${HOME}/<...>/<...>"
AT="@"

# Test the job before actually submitting
#SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch

RUNS_PATH="${SOURCE_PATH}/models/${model}"
echo $RUNS_PATH
mkdir -p $RUNS_PATH

"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%J_slurm.out"
#SBATCH --error="${RUNS_PATH}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="jodahl${AT}kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor|balrog|shelob|smaug"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=6:00:00

echo "Sourcing conda.sh"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
echo "Activating conda environment"
conda activate workshop
nvidia-smi

python disentanglement.py

HERE