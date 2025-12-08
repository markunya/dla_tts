#!/bin/bash
#SBATCH --job-name="dla_tta"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=0-48:00
#SBATCH --output=logs/dla_nv_train_%j.out
#SBATCH --mail-user=mabondarenko_4@edu.hse.ru
#SBATCH --mail-type=ALL
module purge

export PYTHONUNBUFFERED=1
export COMET_API_KEY=3NdmaEabYuPCx2uLXI00rnc0h
python -u train.py -cn=train.yaml writer.run_name=v2_fix_bug
