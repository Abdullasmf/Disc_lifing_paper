#!/bin/bash
#SBATCH --job-name=GPU3(PN)
#SBATCH --output=Disc_lifing_paper/GPU3.log
#SBATCH --error=Disc_lifing_paper/GPU3.log
#SBATCH --time=80:00:00
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

# echo "===============================Zonal-Edge-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Zonal-Edge-Arc-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge_arc/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Zonal-Edge-Arc-_headfeat-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge_arc/PointNetMLPJoint_headfeat/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Zonal-Edge-Arc-Feature-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge_arc_feat/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Zonal-Edge-Arc-Feature-_headfeat-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge_arc_feat/PointNetMLPJoint_headfeat/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Zonal-Edge-No-Stress-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge_no_stress/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Zonal-Edge-Proximity-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge_Prox/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Zonal-Edge-ZoneID-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge_zoneID/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Zonal-Edge-ZoneID-_headfeat-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Zonal/Edge_zoneID/PointNetMLPJoint_headfeat/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# # echo "===============================Zonal-Full-PointNetMLPJoint==============================="
# # python -u Disc_lifing_paper/Zonal/Full/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2

# echo "CHANGING TO UNIFORM DATASET"
# echo "===============================Uniform-Edge-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Uniform/Edge/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Uniform-Edge-Arc-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Uniform/Edge_arc/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Uniform-Edge-Arc-_headfeat-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Uniform/Edge_arc/PointNetMLPJoint_headfeat/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Uniform-Edge-Arc-Feature-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Uniform/Edge_arc_feat/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Uniform-Edge-Arc-Feature-_headfeat-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Uniform/Edge_arc_feat/PointNetMLPJoint_headfeat/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Uniform-Edge-No-Stress-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Uniform/Edge_no_stress/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
echo "===============================Uniform-Edge-Proximity-PointNetMLPJoint==============================="
python -u Disc_lifing_paper/Uniform/Edge_Prox/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2
# echo "===============================Uniform-Full-PointNetMLPJoint==============================="
# python -u Disc_lifing_paper/Uniform/Full/PointNetMLPJoint/GPUL2.py --preset S_full_ln_pos12 --initial-batch 2

echo "DONE"
