#!/usr/bin/env bash
SOURCE_PATH="${HOME}/thesis/betapara"
AT="@"
#SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch
history="${SOURCE_PATH}/history"
mkdir -p history
"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${history}/%J_slurm.out"
#SBATCH --error="${history}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="joadahl${AT}kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor|balrog|shelob|sma$
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60GB
#SBATCH --time=12:00:00

echo "Sourcing conda.sh"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
echo "Activating conda environment"
conda activate workshop
nvidia-smi

python disentanglement.py
HERE

