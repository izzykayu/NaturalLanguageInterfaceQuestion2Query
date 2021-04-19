#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --time=100:00
#SBATCH --mem=20GB
#SBATCH --job-name=Python_job
#SBATCH --mail-type=END
#SBATCH --mail-user=metzgi01@phoenix.nyumc.org


module purge
module load pytorch/python2.7/0.3.0_4
module load pytorch/python2.7/0.3.0_4
module load gcc/6.3.0
pip install torchwordemb --user
 
file_path="/ifs/home/metzgi01/NaturalLanguageInterface/CliNER/data/jun22_train_neg"

for filename in data/jun22_train_neg/*.txt
do
    ./cliner predict --txt $filename --out data/con_neg_val_jun26 --format i2b2 --model models/2012_i2b2_test.model
done
