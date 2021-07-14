# Requirements

This directory contains the Python requirements for the specific experiments. Since experiments are performed on Taurus, which uses a module system, big dependencies like PyTorch are already installed and are therefore commented out in the respective requirement files.

There are different requirements for experiments 1 and 2 because they are executed on different partitions: Because the `ml` partition (which was used for experiment 1) is not able to load a PyTorch version > 1.6, the `Alpha Centauri` partition is used. Pytorch > 1.6 is needed for gpytorch and botorch, which are used in the second experiment to build a Gaussian Process.

`.txt` files are pip requirements, whereas `.yaml` files are anaconda requirements. Anaconda requirement files where not tested for functionality.

## Installation

**If you are using Taurus:** First load the needed modules for the partition you're planning to use:
- _ml (V100 GPUs)_: `module load modenv/ml  torchvision/0.7.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0`
- _alpha centauri (A100 GPUs)_: `module load modenv/hiera  GCCcore/8.3.0  Python/3.7.4`

**If you are _not_ using Taurus:** Uncomment the commented lines in the requirement files.

### Installing PIP Requirements With VENV

Steps to create a new venv environment and install the requirements into it:
- `python -m venv <name>`
- `source <name>/bin/activate` (<name> should now appear in front of your input prompt)
- make sure that `which python` lists your newly created environment
- `python -m pip install -r <requirements_file>`

### Installing Anaconda Requirements With Conda

Steps to create a new conda environment:
- The first line in the `.yaml` files sets the environment name, change it if you like (now assuming \<name>)
- `conda env create -f <environment_file>`
- `conda activate <name>`
