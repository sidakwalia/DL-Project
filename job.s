#!/bin/bash
#
#SBATCH --job-name=resnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIl
#SBATCH --mail-user=mm12318@nyu.edu
#SBATCH --output=slurm_resnet9_%j.out

module purge

singularity exec --nv \
                 --overlay /scratch/mm12318/DL.img \
                /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "/scratch/mm12318/miniconda/envs/DL/bin/python /home/mm12318/DL_Class/DL-Project/train.py"
