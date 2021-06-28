#!/bin/bash
#SBATCH --partition=ml
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M

#SBATCH -J "exp_2_v100_gp_seed_1"

#SBATCH -A p_da_studenten
#SBATCH --mail-user=julien.fischer@mailbox.tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /beegfs/global0/ws/s8732099-diploma_thesis/slurm_output/exp_2/v100_gp_seed_1_normal.out
#SBATCH -e /beegfs/global0/ws/s8732099-diploma_thesis/slurm_output/exp_2/v100_gp_seed_1_error.out

module load modenv/ml
module load Python/3.7.4-GCCcore-8.3.0
source /beegfs/global0/ws/s8732099-diploma_thesis/.venv/gaea_extended/bin/activate
export PYTHONPATH=$PYTHONPATH:/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects

python /beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/cnn/experiments_da.py \
    mode=gp \
    run_search_phase.seed=21 \
    run_eval_phase.seed=12 \
    method.gp_seed=1 \
    run_search_phase.data=/beegfs/global0/ws/s8732099-diploma_thesis/data \
    run_eval_phase.data=/beegfs/global0/ws/s8732099-diploma_thesis/data \
    run_search_phase.autodl=/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects \
    run_eval_phase.autodl=/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects \
    run_search_phase.n_threads_data=0 \
    run_eval_phase.n_threads_data=0 \
    run_eval_phase.number_gpus=6 \
    train_search_phase.batch_size=256 \
    train_eval_phase.batch_size=128 \
    hydra.run.dir=/beegfs/global0/ws/s8732099-diploma_thesis/experiments_da/exp_2/v100/\${method.name}-seed_\${method.gp_seed}

exit 0