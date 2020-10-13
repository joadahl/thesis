#!/usr/bin/env bash
AT="@"


#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%J_slurm.out"
#SBATCH --error="${RUNS_PATH}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="jodahl${AT}kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor|balrog|shelob|smaug"
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --time=12:00:00


python test.py
