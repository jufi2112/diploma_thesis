#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000M
#SBATCH -J "grid_search_sequential_seed_3030"
#SBATCH --mail-user=julien.fischer@mailbox.tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A p_da_studenten
#SBATCH -o /scratch/ws/1/s8732099-da/slurm_output/grid_search_sequential_seed_3030_normal.out
#SBATCH -e /scratch/ws/1/s8732099-da/slurm_output/grid_search_sequential_seed_3030_error.out

module load modenv/hiera GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 PyTorch/1.7.1
source /scratch/ws/1/s8732099-da/.venv/gaea_alpha/bin/activate
export PYTHONPATH=$PYTHONPATH:/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects

python /scratch/ws/1/s8732099-da/git/gaea_release/cnn/experiments_da.py \
    mode=grid_search \
    method.mode=sequential \
    run_search_phase.seed=3030 \
    run_eval_phase.seed=3030 \
    run_search_phase.data=/scratch/ws/1/s8732099-da/data \
    run_eval_phase.data=/scratch/ws/1/s8732099-da/data \
    run_search_phase.autodl=/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects \
    run_eval_phase.autodl=/scratch/ws/1/s8732099-da/git/gaea_release/AutoDL-Projects \
    hydra.run.dir=/scratch/ws/1/s8732099-da/experiments_da/\${method.name}

exit 0