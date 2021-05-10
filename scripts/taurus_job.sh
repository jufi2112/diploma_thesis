#!/bin/bash
#SBATCH --partition=ml
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5772M
#SBATCH -J "grid_search_sequential_seed_2554"
#SBATCH --mail-user=julien.fischer@mailbox.tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A p_da_studenten

module load modenv/ml
module load torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0
source /scratch/ws/1/s8732099-da/.venv/gaea/bin/activate
export PYTHONPATH=$PYTHONPATH:/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects

python /scratch/ws/1/s8732099-da/git/gaea_release/cnn/experiments_da.py \
    mode=grid_search \
    method.mode=sequential \
    run_search_phase.seed=2554 \
    run_eval_phase.seed=2554 \
    run_search_phase.data=/scratch/ws/1/s8732099-da/data \
    run_eval_phase.data=/scratch/ws/1/s8732099-da/data \
    run_search_phase.autodl=/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects \
    run_eval_phase.autodl=/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects \
    hydra.run.dir=/scratch/ws/1/s8732099-da/experiments_da/\${method.name}

exit 0

