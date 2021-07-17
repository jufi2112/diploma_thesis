#!/bin/bash
#SBATCH --partition=ml
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH -J "exp_1_data-shuffle_false_v100_grid_search_search_only_seed_2554"
#SBATCH --mail-user=julien.fischer@mailbox.tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A p_da_studenten
#SBATCH -o /beegfs/global0/ws/s8732099-diploma_thesis/slurm_output/exp_1_data-shuffle_false/v100_grid_search_search_only_seed_2554_normal.out
#SBATCH -e /beegfs/global0/ws/s8732099-diploma_thesis/slurm_output/exp_1_data-shuffle_false/v100_grid_search_search_only_seed_2554_error.out

module load modenv/ml
module load torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
source /home/s8732099/.venv/gaea_exp1_ml/bin/activate
export PYTHONPATH=$PYTHONPATH:/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects

python /beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/cnn/experiments_da.py \
    mode=grid_search \
    method.mode=search_only \
    search.single_level_shuffle=false \
    run_search_phase.seed=2554 \
    run_eval_phase.seed=2554 \
    run_search_phase.data=/beegfs/global0/ws/s8732099-diploma_thesis/data \
    run_eval_phase.data=/beegfs/global0/ws/s8732099-diploma_thesis/data \
    run_search_phase.autodl=/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects \
    run_eval_phase.autodl=/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects \
    run_search_phase.n_threads_data=1 \
    run_eval_phase.n_threads_data=1 \
    train_search_phase.batch_size=256 \
    train_eval_phase.batch_size=128 \
    hydra.run.dir=/beegfs/global0/ws/s8732099-diploma_thesis/experiments_da/exp_1_data-shuffle_false/v100/\${method.name}-batch_size_\${train_search_phase.batch_size}

exit 0