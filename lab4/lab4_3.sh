srun --nodes=1 --tasks-per-node=1 --cpus-per-task=48 --mem=320GB --reservation=ece-gy-9431 --time=00:20:00 --gres=gpu:rtx8000:4 --partition=rtx8000  --pty /bin/bash
module load cuda/11.3.1
module load anaconda3/2020.07
conda activate hpml
cd ~/ECE9143_HPML/lab4
python lab4_3.py --data_path /scratch/yl7897/dataset --num_workers=4 --batch_size=512