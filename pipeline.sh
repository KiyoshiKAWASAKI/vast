#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -N evm_ce_s0

# Required modules
module load conda
conda init bash
source activate vast_evm

python svm_evm_pipeline.py