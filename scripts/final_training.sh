#!/bin/bash

if [ $CONDA_DEFAULT_ENV != "gaea" ]; then
	echo "Activating conda environment 'gaea'"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate gaea
fi
python /home/julienf/git/gaea_release/cnn/train_final.py \
    run.seed=7141 \
    train.drop_path_prob=0.3 \
    run.genotype_path=/home/julienf/data/search-pcdarts-eedarts-cifar10-7141-init_channels-16/genotype.json \
    run.genotype_id=genotype_log_6_tensorboard \
    train.batch_size=64 \
    train.init_channels=36 &> /home/julienf/log_training_2.txt