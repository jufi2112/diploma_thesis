#!/usr/bin/bash
if [ $CONDA_DEFAULT_ENV != "gaea" ]; then
	echo "Activating conda environment 'gaea'"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate gaea
fi
python /home/julienf/git/gaea_release/cnn/train_search.py \
	mode=search_pcdarts \
	nas_algo=eedarts \
	search_config=method_eedarts_space_pcdarts \
	run.seed=2222 \
	run.epochs=50 \
	run.dataset=cifar10 \
	search.single_level=false \
	search.exclude_zero=false &> /home/julienf/log_5.txt
# Logging is already done into ~/data/<search_config>/log.txt but better safe than sorry :D
