#!/bin/bash
#SBATCH --job-name=GPU0(ArGEnT)
#SBATCH --output=Disc_lifing_paper/GPU0.log
#SBATCH --error=Disc_lifing_paper/GPU0.log
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
echo "ARGENT model training!"
echo "loading modules"

. /home/spack/share/spack/setup-env.sh
#spack load py-torch
spack load /j5cepfd
spack load anaconda3

source /usr1/software/miniconda3/etc/profile.d/conda.sh
conda activate /usr1/home/abdulla.fathalla/.aixvipmap/envs/MLEnv

echo "starting script"

echo "===============================Zonal-Edge-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Edge-Arc-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_arc/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Edge-Arc-Feature-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_arc_feat/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Edge-No-Stress-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_no_stress/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Edge-Proximity-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_Prox/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Edge-ZoneID-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_zoneID/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Full-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Full/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2

echo "CHANGING TO UNIFORM DATASET"
echo "===============================Uniform-Edge-ArGEnT==============================="
python -u Disc_lifing_paper/Uniform/Edge/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Uniform-Edge-Arc-ArGEnT==============================="
python -u Disc_lifing_paper/Uniform/Edge_arc/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Uniform-Edge-Arc-Feature-ArGEnT==============================="
python -u Disc_lifing_paper/Uniform/Edge_arc_feat/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Uniform-Edge-No-Stress-ArGEnT==============================="
python -u Disc_lifing_paper/Uniform/Edge_no_stress/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Uniform-Edge-Proximity-ArGEnT==============================="
python -u Disc_lifing_paper/Uniform/Edge_Prox/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Uniform-Full-ArGEnT==============================="
python -u Disc_lifing_paper/Uniform/Full/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2

echo "DONE"
