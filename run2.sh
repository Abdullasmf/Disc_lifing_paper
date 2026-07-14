#!/bin/bash
#SBATCH --job-name=GPU2(PN)
#SBATCH --output=Disc_lifing_paper/GPU2.log
#SBATCH --error=Disc_lifing_paper/GPU2.log
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
echo "PN model training!"
echo "loading modules"

. /home/spack/share/spack/setup-env.sh
#spack load py-torch
spack load /j5cepfd
spack load anaconda3

source /usr1/software/miniconda3/etc/profile.d/conda.sh
conda activate /usr1/home/abdulla.fathalla/.aixvipmap/envs/MLEnv

echo "starting script"


echo "===============================Uniform-Edge-Proximity-PointNetMLPJoint==============================="
python -u Disc_lifing_paper/Uniform/Edge_Prox/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos8 --initial-batch 2


echo "DONE"
