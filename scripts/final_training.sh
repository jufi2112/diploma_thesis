#!/bin/bash

if [ $CONDA_DEFAULT_ENV != "gaea" ]; then
	echo "Activating conda environment 'gaea'"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate gaea
fi
python /home/julienf/git/gaea_release/cnn/train_final.py \
    run.seed=95 \
    train.drop_path_prob=0.3 \
    run.genotype_path=/home/julienf/data/genotype.json \
    run.genotype_id=genotype_log &> /home/julienf/log_training.txt \
    train.batch_size=64 \
    train.init_channels=36