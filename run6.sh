#!/bin/bash
#SBATCH --job-name=GPU6(ArGEnT)
#SBATCH --output=Disc_lifing_paper/GPU6.log
#SBATCH --error=Disc_lifing_paper/GPU6.log
#SBATCH --time=80:00:00
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




echo "===============================Zonal-Edge-10%-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_10/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Edge-25%-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_25/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Edge-50%-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_50/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2
echo "===============================Zonal-Edge-75%-ArGEnT==============================="
python -u Disc_lifing_paper/Zonal/Edge_75/ArGEnT_self_att_noSDF/GPU0.py --preset S --initial-batch 2

echo "DONE"
