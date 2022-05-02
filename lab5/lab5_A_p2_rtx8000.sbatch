#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=hpml_lab5_A_rtx8000
#SBATCH --mail-type=END
#SBATCH --mail-user=yl7897@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load anaconda3/2020.07
module load cuda/11.3.1
eval "$(conda shell.bash hook)"
conda activate hpml
RUNDIR=$HOME/ECE9143_HPML/lab5/run/rtx8000/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
cd $RUNDIR
python $HOME/ECE9143_HPML/lab5/lab5_A_p2.py --data_path /scratch/yl7897/dataset --batch_size=128 --num_workers=4