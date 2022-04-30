srun --time=4:00:00 --mem=16GB --gres=gpu:rtx8000:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=4 --pty /bin/bash
python lab5_A_p2.py --data_path /scratch/yl7897/dataset --batch_size=128