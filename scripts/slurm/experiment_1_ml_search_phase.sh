#!/bin/bash
#SBATCH --partition=ml
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH -J "exp_1_v100_grid_search_search_only_seed_2554"
#SBATCH --mail-user=julien.fischer@mailbox.tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A p_da_studenten
#SBATCH -o /scratch/ws/1/s8732099-da/slurm_output/exp_1_repeating/v100_grid_search_search_only_seed_2554_normal.out
#SBATCH -e /scratch/ws/1/s8732099-da/slurm_output/exp_1_repeating/v100_grid_search_search_only_seed_2554_error.out

module load modenv/ml
module load torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
source /scratch/ws/1/s8732099-da/.venv/gaea/bin/activate
export PYTHONPATH=$PYTHONPATH:/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects

python /scratch/ws/1/s8732099-da/git/gaea_release/cnn/experiments_da.py \
    mode=grid_search \
    method.mode=search_only \
    run_search_phase.seed=2554 \
    run_eval_phase.seed=2554 \
    run_search_phase.data=/scratch/ws/1/s8732099-da/data \
    run_eval_phase.data=/scratch/ws/1/s8732099-da/data \
    run_search_phase.autodl=/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects \
    run_eval_phase.autodl=/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects \
    run_search_phase.n_threads_data=1 \
    run_eval_phase.n_threads_data=1 \
    train_search_phase.batch_size=256 \
    train_eval_phase.batch_size=256 \
    hydra.run.dir=/scratch/ws/1/s8732099-da/experiments_da/exp_1_repeating/v100/\${method.name}-batch_size_\${train_search_phase.batch_size}

exit 0