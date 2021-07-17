#!/bin/bash
#SBATCH --partition=ml
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M

#SBATCH -J "exp_1_grid_search_evaluate_only_constant_channels_seed_2554"

#SBATCH -A p_da_studenten
#SBATCH --mail-user=julien.fischer@mailbox.tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /scratch/ws/1/s8732099-da/slurm_output/exp_1/v100_grid_search_evaluate_only_constant_channels_seed_2554_normal.out
#SBATCH -e /scratch/ws/1/s8732099-da/slurm_output/exp_1/v100_grid_search_evaluate_only_constant_channels_seed_2554_error.out

module load modenv/ml
module load torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
source /scratch/ws/1/s8732099-da/.venv/gaea/bin/activate
export PYTHONPATH=$PYTHONPATH:/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects

# setting hyperparameters for search is necessary because of hydra
python /scratch/ws/1/s8732099-da/git/gaea_release/cnn/experiments_da.py \
    mode=grid_search \
    method.mode=evaluate_only \
    method.use_search_channels_for_evaluation=false \
    run_search_phase.seed=2554 \
    run_eval_phase.seed=2554 \
    run_search_phase.data=/scratch/ws/1/s8732099-da/data \
    run_eval_phase.data=/scratch/ws/1/s8732099-da/data \
    run_search_phase.autodl=/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects \
    run_eval_phase.autodl=/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects \
    run_search_phase.n_threads_data=0 \
    run_eval_phase.n_threads_data=0 \
    run_eval_phase.number_gpus=6 \
    train_search_phase.batch_size=256 \
    train_eval_phase.batch_size=128 \
    hydra.run.dir=/scratch/ws/1/s8732099-da/experiments_da/exp_1_constant_eval_channels/v100/\${method.name}-batch_size_256

exit 0