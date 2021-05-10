#!/usr/bin/bash
# bash script for local pc
if [ $CONDA_DEFAULT_ENV != "gaea" ]; then
        echo "Activating conda environment 'gaea'"
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate gaea
fi

python /home/julienf/git/gaea_release/cnn/experiments_da.py  \
        mode=grid_search \
        method.mode=sequential \
        run_search_phase.seed=2554 \
        run_search_phase.data=/home/julienf/data \
        run_search_phase.autodl=/home/julienf/git/gaea_release/AutoDL-Projects \
        run_eval_phase.seed=2554 \
        run_eval_phase.data=/home/julienf/data \
        run_eval_phase.autodl=/home/julienf/git/gaea_release/AutoDL-Projects \
        hydra.run.dir=/home/julienf/data/experiments_da/\${method.name} &>> /home/julienf/log_experiments_da.txt