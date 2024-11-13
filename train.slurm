#!/bin/bash
#SBATCH --job-name=<Job Name>
#SBATCH --partition=<Partition Name>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:2
#SBATCH --error=%j.err
#SBATCH -output=%j.out

export PATH=/path/to/python/bin:$PATH
export PATH=/path/to/cuda/bin:$PATH
export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/python/lib:$LD_LIBRARY_PATH

python -m torch.distributed.launch --master_port=19253 --nproc_per_node=2 trainUM.py -opt=Configurations/config.yml --launcher pytorch