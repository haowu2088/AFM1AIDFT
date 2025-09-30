#!/bin/bash

#SBATCH --job-name=kramers 
#SBATCH --ntasks=32
#SBATCH --partition=gpu
#SBATCH --nodelist=execute-3001
#SBATCH --mem=40960

source /home/uwm/wu67/Data/wu67/conda/bin/activate

sh pretrain.sh     

sh train.sh     

sh predict.sh
