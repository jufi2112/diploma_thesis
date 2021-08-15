#!/bin/bash
#SBATCH --partition=ml
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH -J "test_set_inference"

#SBATCH -A p_da_studenten
#SBATCH --mail-user=julien.fischer@mailbox.tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /beegfs/global0/ws/s8732099-diploma_thesis/slurm_output/architecture_evaluation/test_set_performance_normal.out
#SBATCH -e /beegfs/global0/ws/s8732099-diploma_thesis/slurm_output/architecture_evaluation/test_set_performance_error.out

module load modenv/ml
module load torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
source /home/s8732099/.venv/gaea_exp1_ml/bin/activate
export PYTHONPATH=$PYTHONPATH:/beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/AutoDL-Projects

# setting hyperparameters for search is necessary because of hydra
python /beegfs/global0/ws/s8732099-diploma_thesis/git/diploma_thesis/cnn/experiments_da_evaluation.py \
    run.data=/beegfs/global0/ws/s8732099-diploma_thesis/data \
    run.checkpoint_path=/beegfs/global0/ws/s8732099-diploma_thesis/best_runs_to_test \
    hydra.run.dir=/beegfs/global0/ws/s8732099-diploma_thesis/experiments_da/test_performance/

exit 0