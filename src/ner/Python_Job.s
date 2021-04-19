#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=100:00
#SBATCH --mem=20GB
#SBATCH --job-name=Python_job
#SBATCH --mail-type=END
#SBATCH --mail-user=im1247@nyu.edu


module purge
module load pytorch/python2.7/0.3.0_4
module load pytorch/python2.7/0.3.0_4
module load gcc/6.3.0
pip install torchwordemb --user

srun python train.py train.out
