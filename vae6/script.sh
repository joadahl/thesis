#!/usr/bin/env bash
SOURCE_PATH="${HOME}/thesis/vae6"
AT="@"
#SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch
slurm="${SOURCE_PATH}/slurm"
#modelstore="${SOURCE_PATH}/modelstore"
mkdir -p slurm
#mkdir -p modelstore
"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${slurm}/%J_slurm.out"
#SBATCH --error="${slurm}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="joadahl${AT}kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor|balrog|shelob|smaug"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60GB
#SBATCH --time=48:00:00

echo "Sourcing conda.sh"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
echo "Activating conda environment"
conda activate workshop
nvidia-smi

python loadmodel.py
HERE

