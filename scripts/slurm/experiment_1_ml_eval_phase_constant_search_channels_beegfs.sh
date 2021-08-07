#!/bin/bash
#SBATCH --partition=ml
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M

#SBATCH -J "exp_1_evaluate_only_constant_search_channels_4_seed_2554_beegfs"

#SBATCH -A p_da_studenten
#SBATCH --mail-user=julien.fischer@mailbox.tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /beegfs/global0/ws/s8732099-diploma_thesis/slurm_output/exp_1_constant_search_channels/v100_grid_search_evaluate_only_constant_search_channels_4_seed_2554_normal.out
#SBATCH -e /beegfs/global0/ws/s8732099-diploma_thesis/slurm_output/exp_1_constant_search_channels/v100_grid_search_evaluate_only_constant_search_channels_4_seed_2554_error.out

zeta_eval=4

module load modenv/ml
module load torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
source /home/s8732099/.venv/gaea_exp1_ml/bin/activate
export PYTHONPATH=$PYTHONPATH:/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects

# setting hyperparameters for search is necessary because of hydra
python /beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/cnn/experiments_da.py \
    mode=grid_search \
    method.mode=evaluate_only \
    method.init_channels_to_check=[24] \
    method.use_search_channels_for_evaluation=false \
    train_eval_phase.init_channels=$zeta_eval \
    run_search_phase.seed=2554 \
    run_eval_phase.seed=2554 \
    run_search_phase.data=/beegfs/global0/ws/s8732099-diploma_thesis/data \
    run_eval_phase.data=/beegfs/global0/ws/s8732099-diploma_thesis/data \
    run_search_phase.autodl=/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects \
    run_eval_phase.autodl=/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects \
    run_search_phase.n_threads_data=0 \
    run_eval_phase.n_threads_data=0 \
    run_eval_phase.number_gpus=6 \
    train_search_phase.batch_size=256 \
    train_eval_phase.batch_size=128 \
    hydra.run.dir=/beegfs/global0/ws/s8732099-diploma_thesis/experiments_da/exp_1_eval_$zeta_eval/v100/\${method.name}-batch_size_256

exit 0