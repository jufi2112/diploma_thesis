# Diploma Thesis
This is the repository for my diploma thesis in computer science at the TU Dresden on "Enhancing Single Step Neural Architecture Search by Two Stage Meta-Parameter Optimization".

## Usage of Existing Code
My work is based on a fork from the repository of [Geometry-Aware Gradient Algorithms for Neural Architecture Search](https://arxiv.org/pdf/2004.07802.pdf) (short _GAEA_) which I extended to allow the tuning of hyperparameters during the search and evaluation phases of single-level neural architecture search (_NAS_). While not explicitely stated, the [GAEA repository](https://github.com/liamcli/gaea_release) uses code from the [PC-DARTS repository](https://github.com/yuhuixu1993/PC-DARTS) which in turn uses code from the original [DARTS repository](https://github.com/quark0/darts). While the original DARTS code is licensed under the Apache License 2.0, _PC-DARTS_ and _GAEA_ are missing a license information. Therefore, this repository uses the Apache License 2.0 of the original DARTS repository.

What follows is the content of the original GAEA readme. When I find time I'm gonna rewrite this to comply with this project. 

# [Geometry-Aware Gradient Algorithms for Neural Architecture Search](https://arxiv.org/pdf/2004.07802.pdf)
This repository contains the code required to run the experiments for the DARTS search space over CIFAR-10 and the NAS-Bench-201 search space over CIFAR-10, CIFAR-100, and ImageNet16-120.  Code to run the experiments on the DARTS search space over ImageNet and the NAS-Bench-1Shot1 search spaces will be made available in forked repos subsequently.  

First build the docker image using the provided docker file:
`docker build -t [name] -f docker/config.dockerfile .`

Then run a container with the image, e.g.:
`docker run -it --gpus all --rm [name]`

Then run the commands below from within the container.  The [scripts](scripts) provided may be helpful.

## DARTS Search Space on CIFAR-10
Search using GAEA PC-DARTS by running
~~~
python train_search.py 
  mode=search_pcdarts 
  nas_algo=eedarts 
  search_config=method_eedarts_space_pcdarts 
  run.seed=[int] 
  run.epochs=50
  run.dataset=cifar10
  search.single_level=false
  search.exclude_zero=false
~~~

Evaluate architecture found in search phase by running
~~~
python train_aws.py
  train.arch=[archname which must be specified in cnn/search_spaces/darts/genotypes.py]
  run.seed=[int]
  train.drop_path_prob=0.3
~~~
  
## NAS-Bench-201 Search Space
Search using GAEA DARTS by running
~~~
python train_search.py
  mode=search_nasbench201
  nas_algo=edarts
  search_config=method_edarts_space_nasbench201
  run.seed=[int]
  run.epochs=25
  run.dataset=[one of cifar10, cifar100, or ImageNet16-120]
  search.single_level=[true for ERM and false for bilevel]
  search.exclude_zero=true
~~~

