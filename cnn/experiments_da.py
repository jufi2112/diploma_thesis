"""Experiments for my diploma thesis 
    "Enhancing Single Step Neural Architecture Search by Two Stage Meta-Parameter Optimization"
"""
import numpy as np
import hydra
import os
import sys
import io
import PIL
import logging
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import namedtuple
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from datetime import timedelta
from timeit import default_timer as timer

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

import train_utils
import visualize
from genotypes_to_visualize import Genotype

from architect.architect_edarts import ArchitectEDARTS as Architect
from search_spaces.pc_darts.model_search import PCDARTSNetwork  # Network for search
from search_spaces.darts.model import NetworkCIFAR              # Network for evaluation

@hydra.main(config_path="../configs/experiments_da/experiments_da.yaml", strict=False)
def main(args):
    """Performs either grid search over init_channels or employs a Gaussian process to search the best value for init_channels.
    Gaussian process is not implemented, yet (tm)
    """
    np.set_printoptions(precision=3)

    working_directory = os.getcwd()
    # TODO: dunno why original implementation used double logging
    log = os.path.join(working_directory, f"overall_log_seed_{args.run_search_phase.seed}.txt")
    train_utils.set_up_logging(log)

    logging.info(f"Hyperparameters: \n{args.pretty()}")

    if args.method.name == "grid_search":
        (
            search_history,
            eval_history,
            runtime_search_phase,
            runtime_eval_phase
        ) = grid_search(args)

        if eval_history is not None:
            # find best performing init_channels
            best_init_channels = list(eval_history)[0]

            encountered_eval_errors = []

            for channels, values in eval_history.items():
                if "search_error" in values.keys() or "eval_error" in values.keys():
                    if "eval_error" in values.keys():
                        encountered_eval_errors.append({
                            'init_channels': channels,
                            'error': values['eval_error']
                        })
                    continue
                if values['valid_acc'] > eval_history[best_init_channels]['valid_acc']:
                    best_init_channels = channels

            # search for search errors
            encountered_search_errors = []
            for channels, values in search_history.items():
                if "error" in values.keys():
                    encountered_search_errors.append({
                        'init_channels': channels,
                        'error': values['error']
                    })

            # print errors
            logging.info(f"Encountered a total of {len(encountered_eval_errors) + len(encountered_search_errors)} errors during search and evaluation.")
            logging.info(f"{len(encountered_search_errors)} erros during architecture search.")
            logging.info(f"{len(encountered_eval_errors)} errors during architecture evaluation.")
            logging.info("NAS errors:")
            for error in encountered_search_errors:
                logging.info(f"| init_channels: {error['init_channels']} | error: {error['error']} |")

            logging.info("Evaluation errors:")
            for error in encountered_eval_errors:
                logging.info(f"| init_channels: {error['init_channels']} | error: {error['error']} |")

            # print information about best performing init_channels
            logging.info(f"Grid search finished after a total of {timedelta(seconds=(runtime_search_phase + runtime_eval_phase))} hh:mm:ss.")
            logging.info(f"NAS took {timedelta(seconds=runtime_search_phase)} hh:mm:ss.")
            logging.info(f"Evaluation of searched architectures took {timedelta(seconds=runtime_eval_phase)} hh:mm:ss.")

            logging.info(f"Best found architecture with init_channels={best_init_channels}: ")
            logging.info(f"Path to checkpoint: {eval_history[best_init_channels]['checkpoint_path']}")
            logging.info(f"Number of model parameters: {eval_history[best_init_channels]['model_parameters']}")
            logging.info(f"Genotype: {search_history[best_init_channels]['best_genotype']}")
            logging.info(f"Evaluation train_acc: {eval_history[best_init_channels]['train_acc']} %")
            logging.info(f"Search train_acc:     {search_history[best_init_channels]['search_train_acc']} %")
            logging.info(f"Evaluation valid_acc: {eval_history[best_init_channels]['valid_acc']} %")
            logging.info(f"Search valid_acc:     {search_history[best_init_channels]['search_valid_acc']} %")
            logging.info(f"Evaluation time: {timedelta(seconds=eval_history[best_init_channels]['overall_eval_time'])} hh:mm:ss.")
            logging.info(f"Search time: {timedelta(seconds=search_history[best_init_channels]['overall_search_time'])} hh:mm.ss.")
            logging.info(f"Max memory allocated during evaluation: {eval_history[best_init_channels]['max_mem_allocated_MB']} MB.")
            logging.info(f"Max memory allocated during search: {search_history[best_init_channels]['max_mem_allocated_MB']} MB.")

        elif search_history is not None:
            logging.info("Performed search phase only, in order to obtain the true performance of a genotype, you still need to evaluate it.")
            

            best_init_channels = list(search_history)[0]
            encountered_errors = []

            for channels, values in search_history.items():
                if "error" in values.keys():
                    encountered_errors.append({
                        'init_channels': channels,
                        'error': values['error']
                    })
                    continue
                if values['search_train_acc'] > search_history[best_init_channels]['search_train_acc']:
                    best_init_channels = channels

            logging.info(f"Encountered a total of {len(encountered_errors)} errors during search:")
            for error in encountered_errors:
                logging.info(f"| init_channels: {error['init_channels']} | error: {error['error']}")

            # print stats
            logging.info("The following best genotype is only selected according to best shared-weights training accuracy!")
            logging.info(f"Overall runtime of the search phase: {timedelta(seconds=runtime_search_phase)} hh:mm:ss.")
            logging.info(f"Best performing init_channels: {best_init_channels}")
            logging.info(f"Best genotype: {search_history[best_init_channels]['best_genotype']}")
            logging.info(f"Search train_acc: {search_history[best_init_channels]['search_train_acc']} %")
            logging.info(f"Search valid_acc: {search_history[best_init_channels]['search_valid_acc']} %")
            logging.info(f"Search time: {timedelta(seconds=search_history[best_init_channels]['overall_search_time'])} hh:mm:ss.")
            logging.info(f"Max memory allocated during search: {search_history[best_init_channels]['max_mem_allocated_MB']} MB.")


        else:
            raise ValueError("Both search_history and eval_history are None. Critical error during grid_search. Consult logs for more info.")
    elif args.method.name == "gaussian_process":
        (
            reason,
            incumbent,
            runtime,
            details
        ) = gaussian_process_search(args)

        logging.info(f"Gaussian Process search finished with the following reason: {reason}")
        logging.info(f"After a search time of {timedelta(seconds=runtime)}, the GP came up with the following incumbent: {incumbent}")

    else:
        raise ValueError("Unrecognized method.")


def gaussian_process_search(args):
    """Searches for good performing learning rate candidates with a Gaussian Process

    Args:
        args (OmegaConf): Arguments

    Returns:
        Exception: The reason why the gaussian process was stopped.
        dict: The incumbent.
        float: Runtime of the GP in seconds.
        dict: Details of the search process.
    """
    # setup logging
    cwd = os.getcwd()
    log = os.path.join(cwd, f"log_gaussian_process_seed_{args.run_search_phase.seed}.txt")
    train_utils.set_up_logging(log)

    logging.info(f"Hyperparameters: \n{args.pretty()}")

    # base directories for search and evaluation phases
    base_dir_search = os.path.join(cwd, f"search_phase_seed_{args.run_search_phase.seed}")
    base_dir_eval = os.path.join(cwd, f"evaluation_phase_seed_{args.run_eval_phase.seed}")

    logging.info(f"Starting Gaussian Process search for the best performing learning rate.")
    current_runtime = 0
    gp_start_time = timer()
    torch.manual_seed(args.method.gp_seed)
    gp_rng = torch.get_rng_state()

    # 1. See if outer-loop checkpoint already exists
    if os.path.isfile(os.path.join(cwd, 'outer_loop.ckpt')):
        (
            learning_rates, 
            valid_errors, 
            incumbent, 
            previous_runtime, 
            number_random_samples,
            details,
            gp_rng
        ) = train_utils.load_gp_outer_loop_checkpoint(cwd)
        torch.set_rng_state(gp_rng)
        iteration = -number_random_samples if valid_errors is None else (valid_errors.shape[0] - number_random_samples)
        pairs_trained = 0 if valid_errors is None else max(valid_errors.shape[0] - number_random_samples, 0)
        logging.info("Found an existing outer-loop checkpoint.")
        logging.info(f"    The GP has already been trained with {pairs_trained} learning rate pairs.")
        logging.info(f"    Runtime of resumed GP: {timedelta(seconds=previous_runtime)} (h:m:s)")
        logging.info(f"    Continuing iteration {iteration}")
        current_runtime += previous_runtime
    else:
        # Sample <args.method.random_samples> values for search and evaluation learning rates at random or use provided values
        logging.info("No outer-loop checkpoint found, starting search from scratch.")
        incumbent = {'lrs': None, 'valid_error': None}
        details = {'search': {}, 'evaluation': {}}
        number_random_samples = args.method.random_samples
        
        # 2. Sample <args.method.random_samples> values for search and evaluation learning rates at random
        if args.method.manual_random_samples is None:
            # No values provided, sample at random
            logging.info(f"No prior learning rates given. Sampling {number_random_samples} learning rate pairs at random.")
            random_lrs_search = torch.FloatTensor(
                number_random_samples, 
                1
            ).uniform_(
                args.method.learning_rate_interval['search'][0] + args.method.interval_epsilon['search'][0],
                args.method.learning_rate_interval['search'][1] - args.method.interval_epsilon['search'][1]
            )
            random_lrs_eval = torch.FloatTensor(
                number_random_samples,
                1
            ).uniform_(
                args.method.learning_rate_interval['evaluation'][0] + args.method.interval_epsilon['evaluation'][0], 
                args.method.learning_rate_interval['evaluation'][1] - args.method.interval_epsilon['evaluation'][1]
            )
            learning_rates = torch.cat((random_lrs_search, random_lrs_eval), dim=1)
            
        else:
            # Values manually provided, fill with random samples if not enough values given
            learning_rates = torch.tensor(args.method.manual_random_samples)
            logging.info(f"{learning_rates.shape[0]} learning rate pairs given as priors.")
            if learning_rates.shape[0] < number_random_samples:
                logging.info(f"Sampling {number_random_samples - learning_rates.shape[0]} pairs at random to reach the configured number of {number_random_samples} priors.")
                random_lrs_search = torch.FloatTensor(
                    number_random_samples - learning_rates.shape[0],
                    1
                ).uniform_(
                    args.method.learning_rate_interval['search'][0] + args.method.interval_epsilon['search'][0],
                    args.method.learning_rate_interval['search'][1] - args.method.interval_epsilon['search'][1]
                )
                random_lrs_eval = torch.FloatTensor(
                    number_random_samples - learning_rates.shape[0],
                    1
                ).uniform_(
                    args.method.learning_rate_interval['evaluation'][0] + args.method.interval_epsilon['evaluation'][0], 
                    args.method.learning_rate_interval['evaluation'][1] - args.method.interval_epsilon['evaluation'][1]
                )
                random_lrs = torch.cat((random_lrs_search, random_lrs_eval), dim=1)
                learning_rates = torch.cat((learning_rates, random_lrs), dim=0)

        # obtain validation errors for learning rate priors
        if args.method.manual_random_samples_results is None:
            valid_errors = None
        else:
            valid_errors = torch.tensor(args.method.manual_random_samples_results)

        current_runtime += timer() - gp_start_time
        gp_start_time = timer()
        train_utils.save_gp_outer_loop_checkpoint(
            cwd,
            learning_rates,
            valid_errors,
            incumbent,
            current_runtime,
            number_random_samples,
            details,
            torch.get_rng_state()
        )

    try:
        # Main loop
        while True:
            ##############################################################################################################
            ## Loop Logic:                                                                                              ##
            ##     1. Make sure that every learning rate pair has a corresponding validation error                      ##
            ##              - if this is not the case, infer the validation error for all unmatched learning rate pairs ##
            ##     2. Use all learning rate pairs and corresponding validation errors to train a GP                     ##
            ##     3. Use this GP to predict the next candidate learning rate pair that should be tested                ##
            ##              - add this pair to the list of all learning rates                                           ##
            ##     4. Repeat from 1. (until budget exhausted)                                                           ##
            ##     5. ???                                                                                               ##
            ##     6. Profit                                                                                            ##
            ##############################################################################################################

            # Make sure that for every prior learning rate pair, the corresponding validation error is known
            while valid_errors is None or valid_errors.shape[0] < learning_rates.shape[0]:
                if valid_errors is None or (valid_errors.shape[0] - number_random_samples) < 0:
                    iteration = -number_random_samples if valid_errors is None else (valid_errors.shape[0] - number_random_samples)
                    logging.info(f"Starting interation {iteration}.")
                # Perform search + evaluation
                lr_index = 0 if valid_errors is None else valid_errors.shape[0]
                lr_search = learning_rates[lr_index, 0].item()
                lr_eval = learning_rates[lr_index, 1].item()
                if iteration < 0:
                    logging.info(f"Evaluating random samples, no GP candidate is used: \nSearch lr={lr_search}\nEval lr={lr_eval}")
                else:
                    logging.info(f"Evaluating candidate learning rates:\nSearch lr={lr_search}\nEval lr={lr_eval}")

                # search phase
                args.run = args.run_search_phase
                args.train = args.train_search_phase
                args.train.learning_rate = lr_search

                # check if search with same learning rate was already performed
                possible_genotype_file = os.path.join(base_dir_search, 'genotypes', f'genotype_learning_rate_{lr_search}.json')
                if os.path.isfile(possible_genotype_file):
                    # evaluation considers string to be a path to the genotype file and all other types to be the genotype itself
                    best_genotype = possible_genotype_file
                    try:
                        search_results = details['search'][lr_search]
                    except KeyError:
                        logging.info(f"KeyError while trying to access the details of search performed with learning rate {lr_search}.")
                        search_results = None
                    logging.info("Search with the specified learning rate has already been performed. Skipping...")
                elif lr_search in details['search'].keys() and details['search'][lr_search] is not None and not issubclass(details['search'][lr_search], Exception):
                    search_results = details['search'][lr_search]
                    best_genotype = search_results['best_genotype']
                    logging.info("Search with the specified learning rate has already been performed. Skipping...")
                else:
                    logging.info("Performing search phase...")
                    current_runtime += timer() - gp_start_time
                    gp_rng = torch.get_rng_state()
                    torch.cuda.empty_cache()
                    try:
                        (
                            best_genotype,
                            best_genotype_search_time,
                            search_train_acc,
                            search_valid_acc,
                            single_search_time,
                            max_mem_allocated_MB,
                            max_mem_reserved_MB
                        ) = search_phase(args, base_dir_search, 'learning_rate')

                        torch.set_rng_state(gp_rng)
                        current_runtime += single_search_time
                        gp_start_time = timer()

                        search_results = {
                            'best_genotype': best_genotype,
                            'best_genotype_search_time': best_genotype_search_time,
                            'train_acc': search_train_acc,
                            'valid_acc': search_valid_acc,
                            'runtime': single_search_time,
                            'max_mem_allocated_MB': max_mem_allocated_MB,
                            'max_mem_reserved_MB': max_mem_reserved_MB
                        }

                        details['search'][lr_search] = search_results

                    except Exception as e:
                        torch.set_rng_state(gp_rng)
                        logging.info(f"Encountered the following exception during search: {e}")
                        gp_start_time = timer()
                        if lr_search in details['search'].keys() and details['search'][lr_search] is not None and issubclass(details['search'][lr_search], Exception):
                            logging.info(f"Search for the given learning rate failed 2 times, removing this pair from the priors...")
                            learning_rates = torch.cat((learning_rates[:lr_index], learning_rates[lr_index+1:]), dim=0)
                        else:
                            details['search'][lr_search] = e
                        continue

                    current_runtime += timer() - gp_start_time
                    gp_start_time = timer()
                    train_utils.save_gp_outer_loop_checkpoint(
                        cwd,
                        learning_rates,
                        valid_errors,
                        incumbent,
                        current_runtime,
                        number_random_samples,
                        details,
                        torch.get_rng_state()
                    )
                    logging.info(f"Search phase finished after {timedelta(seconds=details['search'][lr_search]['runtime'])}")

                # evaluation phase
                logging.info(f"Performing evaluation of found genotype: {best_genotype}")
                args.run = args.run_eval_phase
                args.train = args.train_eval_phase
                args.train.learning_rate = lr_eval

                smp = mp.get_context('spawn')
                result_queue = smp.Queue()
                current_runtime += timer() - gp_start_time
                gp_rng = torch.get_rng_state()
                torch.cuda.empty_cache()
                try:
                    os.environ['MASTER_ADDR'] = 'localhost'
                    os.environ['MASTER_PORT'] = train_utils.find_free_port()
                    mp.spawn(
                        evaluation_phase,
                        args=(
                            args,
                            base_dir_eval,
                            'learning_rate',
                            best_genotype,
                            result_queue,
                            lr_search
                        ),
                        nprocs=args.run.number_gpus
                    )
                    logging.info("Evaluation phase completed successfully")
                    torch.set_rng_state(gp_rng)
                    result = result_queue.get()
                    current_runtime += result['overall_runtime']
                    gp_start_time = timer()
                    val_err = 100 - torch.tensor([[result['valid_acc_best_observed']]])
                    details['evaluation'][f"{lr_search}_{lr_eval}"] = result
                    del result
                    logging.info(f"Evaluation phase finished after {timedelta(seconds=details['evaluation'][f'{lr_search}_{lr_eval}']['overall_runtime'])}")
                    logging.info(f"Validation error: {val_err[0].item()}")

                except Exception as e:
                    torch.set_rng_state(gp_rng)
                    logging.info(f"Encountered the following exception during evaluation: {e}")
                    gp_start_time = timer()
                    if f"{lr_search}_{lr_eval}" in details['evaluation'].keys() and details['evaluation'][f'{lr_search}_{lr_eval}'] is not None and issubclass(details['evaluation'][f'{lr_search}_{lr_eval}'], Exception):
                        logging.info(f"Evaluation phase for the given learning rate failed 2 times, removing this pair from the priors...")
                        learning_rates = torch.cat((learning_rates[:lr_index], learning_rates[lr_index+1:]), dim=0)
                    else:
                        details['evaluation'][f"{lr_search}_{lr_eval}"] = e
                    continue
                    
                valid_errors = val_err if valid_errors is None else torch.cat((valid_errors, val_err), dim=0)

                # update incumbent
                incumbent = train_utils.determine_incumbent(learning_rates, valid_errors)
                logging.info(f"Current incumbent: {incumbent}")
                
                current_runtime += timer() - gp_start_time
                logging.info(f"Current runtime of the GP search: {timedelta(seconds=current_runtime)}")
                gp_start_time = timer()
                train_utils.save_gp_outer_loop_checkpoint(
                    cwd,
                    learning_rates,
                    valid_errors,
                    incumbent,
                    current_runtime,
                    number_random_samples,
                    details,
                    torch.get_rng_state()
                )      
        
            # same number of learning rate pairs and corresponding validation errors --> Perform GP
                
            iteration = valid_errors.shape[0] - number_random_samples
            logging.info(f"Starting iteration {iteration}.")

            # Create GP
            gp = SingleTaskGP(learning_rates, valid_errors)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            # Acquisition function
            EI = ExpectedImprovement(gp, best_f=incumbent['valid_error'], maximize=False)
            bounds = torch.tensor(
                [
                    [
                        args.method.learning_rate_interval['search'][0] + args.method.interval_epsilon['search'][0],
                        args.method.learning_rate_interval['evaluation'][0] + args.method.interval_epsilon['evaluation'][0]
                    ],
                    [
                        args.method.learning_rate_interval['search'][1] - args.method.interval_epsilon['search'][1],
                        args.method.learning_rate_interval['evaluation'][1] - args.method.interval_epsilon['evaluation'][1]
                    ]
                ]
            )

            # Optimize acquisition function
            candidate, acq_value = optimize_acqf(
                EI,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=20
            )

            logging.info(f"Learning rate candidate is {candidate}")

            # add candidate to learning_rates so that it gets evaluated in the next iteration
            learning_rates = torch.cat((learning_rates, candidate), dim=0)

            current_runtime += timer() - gp_start_time
            gp_start_time = timer()
            train_utils.save_gp_outer_loop_checkpoint(
                cwd,
                learning_rates,
                valid_errors,
                incumbent,
                current_runtime,
                number_random_samples,
                details,
                torch.get_rng_state()
            )

    except Exception as e:
        return e, incumbent, current_runtime, details


def grid_search(args):
    """Performs grid search

    Args:
        args (OmegaConf): Arguments

    Returns:
        dict: Search history.
        dict: Evaluation history.
        float: Overall runtime of the search phase in seconds.
        float: Overall runtime of the evaluation phase in seconds.
    """
    # sequential grid search is no longer supported, since evaluating with multiple GPUs while only using 1 for search is not efficient
    if args.method.mode == "sequential":
        raise ValueError(
            (
                "Sequential grid search is no longer supported, because we use multiple GPUs for the evaluation phase",
                " but only a single GPU during the search phase."
            )
        )
    cwd = os.getcwd()
    log = os.path.join(cwd, f"log_grid_search_{args.method.mode}_seed_{args.run_search_phase.seed}.txt")
    train_utils.set_up_logging(log)

    logging.info(f"Hyperparameters: \n{args.pretty()}")

    #if not torch.cuda.is_available():
    #    logging.error("No GPU device available")
    #    sys.exit(-1)
    #torch.backends.cudnn.benchmark=True

    # overall directory (cwd) is set up by hydra
    # base directories for search and evaluation phases
    base_dir_search = os.path.join(cwd, "search_phase_seed_" + str(args.run_search_phase.seed))
    base_dir_eval = os.path.join(cwd, "evaluation_phase_seed_" + str(args.run_eval_phase.seed))

    logging.info(f"Starting grid search with mode: {args.method.mode}")
    logging.info(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"    Cuda device {i}: {torch.cuda.get_device_name(i)}")

    overall_runtime_search_phase = 0.0  # measures the time spend search
    overall_runtime_eval_phase = 0.0    # measures the time spend evaluating

    # can have mode 'search_only'   - perform only search phase
    #               'evaluate_only' - perform only evaluation phase
    #               'sequential'    - perform search phase followed by evaluation phase <-- not supported any longer
    if args.method.mode in ["search_only", "sequential"]:
        # Create base folders if they don't exist
        os.makedirs(base_dir_search, exist_ok=True)
        if args.method.mode == "sequential":
            os.makedirs(base_dir_eval, exist_ok=True)

        # try to load previous grid search checkpoints (search phase and, if needed, evaluation phase)
        try:
            search_history, previous_runtime_search_phase = train_utils.load_outer_loop_checkpoint(base_dir_search)
            overall_runtime_search_phase += previous_runtime_search_phase
            logging.info(f"Resumed previous grid search search phase checkpoint which already ran for {timedelta(seconds=previous_runtime_search_phase)} hh:mm:ss.")
        except Exception as e:
            logging.info(e)
            logging.info("Starting search phase from scratch.")
            search_history = {}

        if args.method.mode == "sequential":
            try:
                eval_history, previous_runtime_eval_phase = train_utils.load_outer_loop_checkpoint(base_dir_eval)
                overall_runtime_eval_phase += previous_runtime_eval_phase
                logging.info(f"Resumed previous grid search evaluation phase checkpoint which already ran for {timedelta(seconds=previous_runtime_eval_phase)} hh:mm:ss.")
            except Exception as e:
                logging.info(e)
                logging.info("Starting evaluation phase from scratch.")
                eval_history = {}
        else:
            eval_history = None

        # grid search loop
        for step, current_init_channels in enumerate(args.method.init_channels_to_check):
            logging.info(f"| Step {(step+1):3d} / {len(args.method.init_channels_to_check)} |")
            # check if it already exists in search_history (would mean we already performed search for it)
            if current_init_channels in search_history.keys():
                logging.info(f"init_channels={current_init_channels} already in search history. Skipping...")
            else:
                logging.info(f"init_channels={current_init_channels} not yet seen. Starting single search phase...")
                # load correct run and train configuration
                args.run = args.run_search_phase
                args.train = args.train_search_phase
                # set init_channels that should be utilized for search
                args.train.init_channels = current_init_channels
                try:
                    (
                        best_genotype,
                        best_genotype_search_time,
                        search_train_acc,
                        search_valid_acc,
                        single_search_time,
                        max_mem_allocated_MB,
                        max_mem_reserved_MB
                    ) = search_phase(args, base_dir_search, 'init_channels')
                except Exception as e:
                    logging.info(f"Encountered the following exception during search: {e}")
                    logging.info("Continuing with next step")
                    search_history[current_init_channels] = {
                        'error': e
                    }
                    if eval_history is not None:
                        if current_init_channels not in eval_history.keys():
                            eval_history[current_init_channels] = {
                                'search_error': e
                            }
                    # save checkpoint
                    train_utils.save_outer_loop_checkpoint(
                        base_dir_search,
                        search_history,
                        overall_runtime_search_phase
                    )
                    if eval_history is not None:
                        train_utils.save_outer_loop_checkpoint(
                            base_dir_eval,
                            eval_history,
                            overall_runtime_eval_phase
                        )
                    continue

                logging.info(f"Single search finished after a total of {timedelta(seconds=single_search_time)} hh:mm:ss")
                overall_runtime_search_phase += single_search_time

                # write to history
                search_history[current_init_channels] = {
                    'best_genotype': best_genotype,
                    'best_genotype_search_time': best_genotype_search_time,
                    'search_train_acc': search_train_acc,
                    'search_valid_acc': search_valid_acc,
                    'overall_search_time': single_search_time,
                    'max_mem_allocated_MB': max_mem_allocated_MB,
                    'max_mem_reserved_MB': max_mem_reserved_MB
                }

                # save checkpoint
                train_utils.save_outer_loop_checkpoint(
                    base_dir_search,
                    search_history,
                    overall_runtime_search_phase
                )

            if args.method.mode == "search_only":
                continue

            raise ValueError("This code is not supported anymore and will be deleted")
            # evaluate the obtained genotype
            if current_init_channels in eval_history.keys():
                logging.info(f"init_channels={current_init_channels} already in evaluation history. Skipping...")
            else:
                logging.info(f"init_channels={current_init_channels} not yet seen. Starting evaluation phase...")
                # load correct run and train configuration
                args.run = args.run_eval_phase
                args.train = args.train_eval_phase
                if args.method.use_search_channels_for_evaluation:
                    args.train.init_channels = current_init_channels
                
                try:
                    (
                        checkpoint_path,
                        best_weights_train_time,
                        best_weights_train_acc,
                        best_weights_valid_acc,
                        single_training_time,
                        max_mem_allocated_MB,
                        max_mem_reserved_MB,
                        total_params
                    ) = evaluation_phase(
                        args,
                        base_dir_eval,
                        'init_channels',
                        search_history[current_init_channels]['best_genotype'],
                        None,
                        current_init_channels
                    )
                except Exception as e:
                    logging.info(f"Encountered the following exception during evaluation: {e}")
                    logging.info("Continuing with next step.")
                    eval_history[current_init_channels] = {
                        'eval_error': e
                    }
                    # save checkpoint
                    train_utils.save_outer_loop_checkpoint(
                        base_dir_eval,
                        eval_history,
                        overall_runtime_eval_phase
                    )
                    continue

                logging.info(f"Evaluation of genotype finished after a total of {timedelta(seconds=single_training_time)} hh:mm:ss.")
                overall_runtime_eval_phase += single_training_time

                # write to history
                eval_history[current_init_channels] = {
                    'checkpoint_path': checkpoint_path,
                    'best_weights_train_time': best_weights_train_time,
                    'train_acc': best_weights_train_acc,
                    'valid_acc': best_weights_valid_acc,
                    'overall_eval_time': single_training_time,
                    'max_mem_allocated_MB': max_mem_allocated_MB,
                    'max_mem_reserved_MB': max_mem_reserved_MB,
                    'model_parameters': total_params
                }

                # save checkpoint
                train_utils.save_outer_loop_checkpoint(
                    base_dir_eval,
                    eval_history,
                    overall_runtime_eval_phase
                )
            
        # end of loop
        # remove log handler from logging
        logging.getLogger().removeHandler(logging.getLogger().handlers[-1])
        return search_history, eval_history, overall_runtime_search_phase, overall_runtime_eval_phase
    else:
        # evaluate_only
        args.run = args.run_eval_phase
        args.train = args.train_eval_phase
        # try to load previous checkpoint of evaluation history
        try:
            eval_history, runtime_eval = train_utils.load_outer_loop_checkpoint(base_dir_eval)
            overall_runtime_eval_phase += runtime_eval
            logging.info(f"Resumed previous grid search evaluation phase checkpoint which already ran for {timedelta(seconds=runtime_eval)} hh:mm:ss.")
        except Exception as e:
            logging.info(e)
            logging.info("Starting evaluation phase from scratch.")
            eval_history = {}

        # try to load search history that should get evaluated
        try:
            search_history, runtime_search = train_utils.load_outer_loop_checkpoint(base_dir_search)
            overall_runtime_search_phase += runtime_search
            logging.info(f"Successfully loaded search history to evaluate.")
        except Exception as e:
            logging.info(e)
            logging.info("Could not load search history. Please make sure that you have already performed the search phase. Aborting...")
            search_history = None
            return search_history, eval_history, overall_runtime_search_phase, overall_runtime_eval_phase

        # evaluation loop
        for step, current_init_channels in enumerate(args.method.init_channels_to_check):#enumerate(search_history.keys()):
            #logging.info(f"| Step {(step+1)} / {len(search_history.keys())} |")
            logging.info(f"| Step {(step+1)} / {len(args.method.init_channels_to_check)} |")
            # check if the desired init_channels value was already searched
            if current_init_channels not in search_history.keys():
                logging.info(f"init_channels={current_init_channels} not in search history. Please start the search phase first. Skipping...")
                continue
            # check if it already exists in eval_history (would mean we already evaluated it)
            if current_init_channels in eval_history.keys():
                logging.info(f"init_channels={current_init_channels} already in evaluation history. Skipping...")
                continue
            
            if "error" in search_history[current_init_channels].keys():
                logging.info(f"The search history for init_channels={current_init_channels} contains the following error: {search_history[current_init_channels]['error']}")
                logging.info(f"Skipping evaluation of this search result.")
                eval_history[current_init_channels] = {
                    'search_error': search_history[current_init_channels]['error']
                }
                # save checkpoint
                train_utils.save_outer_loop_checkpoint(
                    base_dir_eval,
                    eval_history,
                    overall_runtime_eval_phase
                )
                continue

            logging.info(f"init_channels={current_init_channels} not yet evaluated. Starting evaluation...")
            if args.method.use_search_channels_for_evaluation:
                args.train.init_channels = current_init_channels
            else:
                args.train.init_channels = args.train_eval_phase.init_channels
            smp = mp.get_context('spawn')
            result_queue = smp.Queue()
            #result_queue = mp.Queue()
            
            try:
                #(
                #    checkpoint_path,
                #    best_weights_train_time,
                #    best_weights_train_acc,
                #    best_weights_valid_acc,
                #    single_training_time,
                #    max_mem_allocated_MB,
                #    max_mem_reserved_MB,
                #    total_params
                #) = 
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = train_utils.find_free_port()
                mp.spawn(
                    evaluation_phase,
                    args=(
                        args, 
                        base_dir_eval,
                        'init_channels', 
                        search_history[current_init_channels]['best_genotype'], 
                        result_queue,
                        current_init_channels
                    ),
                    nprocs=args.run.number_gpus
                )
                logging.info("Evaluation function completed successfully")
                result = result_queue.get()
                checkpoint_path = result['checkpoint_path']
                best_weights_train_time = result['runtime_best_observed']
                best_weights_train_acc = result['train_acc_best_observed']
                best_weights_valid_acc = result['valid_acc_best_observed']
                single_training_time = result['overall_runtime']
                max_mem_allocated_MB = result['max_mem_allocated_MB']
                max_mem_reserved_MB = result['max_mem_reserved_MB']
                total_params = result['total_params']
                del result  # might not be necessary since this should not contain a shared-memory tensor
                #evaluation_phase(
                #    args,
                #    base_dir_eval,
                #    current_init_channels,
                #    search_history[current_init_channels]['best_genotype']
                #)
            except Exception as e:
                logging.info(f"Encountered the following exception during evaluation: {e}")
                logging.info("Continuing with next step.")
                eval_history[current_init_channels] = {
                    'eval_error': e
                }
                # save checkpoint
                train_utils.save_outer_loop_checkpoint(
                    base_dir_eval,
                    eval_history,
                    overall_runtime_eval_phase
                )
                continue

            logging.info(f"Evaluation of genotype finished after a total of {timedelta(seconds=single_training_time)} hh:mm:ss.")
            overall_runtime_eval_phase += single_training_time

            # write to history
            eval_history[current_init_channels] = {
                'checkpoint_path': checkpoint_path,
                'best_weights_train_time': best_weights_train_time,
                'train_acc': best_weights_train_acc,
                'valid_acc': best_weights_valid_acc,
                'overall_eval_time': single_training_time,
                'max_mem_allocated_MB': max_mem_allocated_MB,
                'max_mem_reserved_MB': max_mem_reserved_MB,
                'model_parameters': total_params
            }

            # save checkpoint
            train_utils.save_outer_loop_checkpoint(
                base_dir_eval,
                eval_history,
                overall_runtime_eval_phase
            )

        # end of loop
        # remove log handler from logging
        logging.getLogger().removeHandler(logging.getLogger().handlers[-1])
        return search_history, eval_history, overall_runtime_search_phase, overall_runtime_eval_phase


def evaluation_phase(rank, args, base_dir, run_id, genotype_to_evaluate, result_queue, search_phase_id=None):
    """Fully trains a provided genotype
    Code is mostly copied from train_final.py but modified to work for my experiments.
    Best weights are selected according to validation accuracy.

    Args:
        rank (int): Rank for multi-processing. Corresponds to the GPU that should be utilized
        args (OmegaConf): Arguments
        base_dir (str): Path to the base directory the evaluation phase should work in.
        run_id (str): Identifier to distinguish multiple runs.
        genotype_to_evaluate (Genotype or str): Genotype that should be evaluated.
            A string is interpreted as path to a json file that contains the genotype that should be evaluated.
        result_queue (torch.multiprocessing.Queue): Queue where the results of the evaluation phase should be stored to.
        search_phase_id (any): Since evaluation with the same hyperparameter can be carried out for different search phases
            with different results, this ID is used to differentiate between evaluation phases for different search phases.
                For experiment 1 (grid search over initial channels), provide the init_channels used during search.
                For experiment 2 (learning rates), provide the search phase learning rate.
            If no value is provided, this parameter is ignored and multiple evaluation runs with the same hyperparmameter will overwrite each other.
    Returns:
        str: Path to checkpoint that contains the best weights.
        float: Runtime in seconds after which the best weights where found.
        float: Train accuracy of best weights.
        float: Validation accuracy of best weights.
        float: Overall training time in seconds.
        float: Maximum memory allocated in MB.
        float: Maximum memory reserved in MB.
        int: Number of model parameters.
    """
    world_size = args.run.number_gpus
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    # Create folder structure
    if rank == 0:
        try:
            if run_id == 'init_channels' and search_phase_id:
                run_identifier = f"{run_id}_{str(search_phase_id)}" 
            elif run_id == 'learning_rate' and search_phase_id:
                run_identifier = f"{run_id}_search-{search_phase_id}_eval-{args.train[run_id]}"
            else:
                run_identifier = f"{run_id}_{args.train[run_id]}"

        except KeyError:
            run_identifier = f"invalid_run_id_{run_id}"
        log_dir = os.path.join(base_dir, "logs")
        tensorboard_dir = os.path.join(base_dir, "tensorboard")
        checkpoint_dir = os.path.join(base_dir, "checkpoints", f"checkpoint_{run_identifier}")

        for directory in [log_dir, tensorboard_dir, checkpoint_dir]:
            os.makedirs(directory, exist_ok=True)

        # Log file for the current evaluation phase
        logfile = os.path.join(log_dir, f"log_{run_identifier}.txt")
        train_utils.set_up_logging(logfile)

        logging.info(f"Hyperparameters: \n{args.pretty()}")

        # Tensorboard SummaryWriter setup
        tensorboard_writer_dir = os.path.join(tensorboard_dir, run_identifier)
        writer = SummaryWriter(tensorboard_writer_dir)
    #    dist.barrier()
    #else:
    #    dist.barrier()


    #if not torch.cuda.is_available():
    #    logging.error("No GPU device available!")
    #    sys.exit(-1)
    #torch.cuda.set_device(args.run.gpu)
    #torch.backends.cudnn.benchmark=True
    #if type(args.run.gpu) == int:
    #    current_device = torch.cuda.current_device()
    #    logging.info(f"Current cuda device: {current_device} - {torch.cuda.get_device_name(current_device)}")
    #elif type(args.run.gpu) == list:
    #    logging.info(f"Specified to use {len(args.run.gpu)} GPUs.")
    #    logging.info(f"Got: {torch.cuda.device_count()} GPUs.")
    #    for i in range(torch.cude.device_count()):
    #        logging.info(f"    Cuda device {i}: {torch.cuda.get_device_name(torch.cuda.device(i))}")

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    if rank == 0:
        current_device = torch.cuda.current_device()
        logging.info(f"Current cuda device: {current_device} - {torch.cuda.get_device_name(current_device)}")
    args.run.gpu = rank
    torch.backends.cudnn.benchmark=True
    # reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Set random seeds for random, numpy, torch and cuda
    rng_seed = train_utils.RNGSeed(args.run.seed)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(rank)

    # Load datasets
    #num_classes, (train_queue, _), valid_queue, test_queue, (num_train, num_valid, num_test) = train_utils.create_cifar10_data_queues_own(
    #    args, evaluation_mode=True
    #)
    # Manually create datasets so that we can manually split train and validation data and use them with DistributedSampler
    train_dataset, valid_dataset, test_dataset = train_utils.get_cifar10_data_sets(args)
    train_valid_split = int(np.floor(len(train_dataset) * args.train.train_portion))
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(train_valid_split))
    valid_dataset = torch.utils.data.Subset(valid_dataset, np.arange(train_valid_split, len(valid_dataset)))
    num_classes = 10
    num_train = len(train_dataset)
    num_valid = len(valid_dataset)
    num_test = len(test_dataset)
    del test_dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_queue = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train.batch_size,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    valid_queue = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.train.batch_size,
        sampler=valid_sampler,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    if rank == 0:
        #valid_sampler = torch.utils.data.RandomSampler(valid_dataset)
        #valid_queue = torch.utils.data.DataLoader(
        #    dataset=valid_dataset,
        #    batch_size=args.train.batch_size,
        #    sampler=valid_sampler,
        #    pin_memory=True,
        #    num_workers=0
        #)

        logging.info(f"Dataset: {args.run.dataset}")
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Number of training images: {num_train}")
        logging.info(f"Number of validation images: {num_valid}")
        logging.info(f"Number of test images: {num_test}")

    # Load genotype
    if type(genotype_to_evaluate) == str:
        with open(genotype_to_evaluate, 'r') as genotype_file:
            genotype_dict = json.load(genotype_file)
        genotype_to_evaluate = train_utils.dict_to_genotype(genotype_dict)

    # Visualize genotype
    if rank == 0:
        genotype_graph_normal = visualize.plot(genotype_to_evaluate.normal, "", return_type="graph", output_format="png")
        binary_normal = genotype_graph_normal.pipe()
        stream_normal = io.BytesIO(binary_normal)
        graph_normal = np.array(PIL.Image.open(stream_normal).convert("RGB"))
        writer.add_image("Normal_Cell", graph_normal, dataformats="HWC")
        genotype_graph_reduce = visualize.plot(genotype_to_evaluate.reduce, "", return_type="graph", output_format="png")
        binary_reduce = genotype_graph_reduce.pipe()
        stream_reduce = io.BytesIO(binary_reduce)
        graph_reduce = np.array(PIL.Image.open(stream_reduce).convert("RGB"))
        writer.add_image("Reduce_Cell", graph_reduce, dataformats="HWC")
        del genotype_graph_normal
        del binary_normal
        del stream_normal
        del graph_normal
        del genotype_graph_reduce
        del binary_reduce
        del stream_reduce
        del graph_reduce

    # Create model
    model = NetworkCIFAR(
        args.train.init_channels,
        num_classes,
        args.train.layers,
        args.train.auxiliary,
        genotype_to_evaluate
    )
    model = model.cuda(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Converts BatchNorm layers to synced batchnorm layers
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank]
    )

    if rank == 0:
        logging.info(f"Size of model parameters: {train_utils.count_parameters_in_MB(model)} MB")
        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info(f"Total parameters of model: {total_params}")

    optimizer, scheduler = train_utils.setup_optimizer(model, args, len(train_queue))

    # if rank == 0:
    #     logging.info("Entering barrier in front of checkpoint loading")
    # dist.barrier()
    # if rank == 0:
    #     logging.info("Exited barrier")

    # Check if we've already trained
    try:
        (
            start_epochs,
            _,
            previous_runtime,
            best_observed,
            global_peak_mem_allocated_MB,
            global_peak_mem_reserved_MB
        ) = train_utils.load(
            checkpoint_dir,
            rng_seed,
            model,
            optimizer,
            s3_bucket=None,
            gpu=rank
        )
        # if rank == 0:
        #     logging.info("Entering barrier after successfully loading checkpoint")
        # dist.barrier()
        # if rank == 0:
        #     logging.info("Exited barrier")
        if best_observed is None:
            best_observed = {
                "train": 0.0,           # train accuracy of best epoch
                "valid": 0.0,           # validation accuracy of best epoch
                "epoch": 0,             # epoch the best accuracy was observed
                "genotype_raw": None,   # not used, but needs to be in the dict for compatibility with search phase
                "genotype_dict": None,  # not used, but needs to be in the dict for compatibility with search phase
                "runtime": 0.0          # runtime after which the best epoch was observed
            }
        scheduler.last_epoch = ((start_epochs - 1) * len(train_queue)) if args.train.scheduler == "cosine_mgpu" else (start_epochs - 1)
        if rank == 0:
            logging.info(
                (
                    f"Resumed training from a previous checkpoint which was already trained for {start_epochs} epochs and "
                    f"already ran for {timedelta(seconds=previous_runtime)} hh:mm:ss."
                )
            )
            logging.info("This is included in the final runtime report.")
    except Exception as e:
        # if rank == 0:
        #     logging.info("Entering barrier after checkpoint loading threw an exception")
        # dist.barrier()
        # if rank == 0:
        #     logging.info("Exited barrier")
        if rank == 0:
            logging.info(e)
        start_epochs = 0
        previous_runtime = 0
        global_peak_mem_allocated_MB = 0.
        global_peak_mem_reserved_MB = 0.

        best_observed = {
            "train": 0.0,           # train accuracy of best epoch
            "valid": 0.0,           # validation accuracy of best epoch
            "epoch": 0,             # epoch the best accuracy was observed
            "genotype_raw": None,   # not used, but needs to be in the dict for compatibility with search phase
            "genotype_dict": None,  # not used, but needs to be in the dict for compatibility with search phase
            "runtime": 0.0          # runtime after which the best epoch was observed
        }

    if rank == 0:
        logging.info(f"Evaluation phase started for genotype: \n{genotype_to_evaluate}")
        if search_phase_id:
            logging.info(f"The genotype was searched with {run_id} = {search_phase_id}")
        train_start_time = timer()

    # Train loop
    for epoch in range(start_epochs, args.run.epochs):
        train_queue.sampler.set_epoch(epoch)
        valid_queue.sampler.set_epoch(epoch)
        if rank == 0:
            logging.info(f"| Epoch: {epoch:4d}/{args.run.epochs} | lr: {scheduler.get_last_lr()[0]} |")
        model.drop_path_prob = args.train.drop_path_prob * epoch / args.run.epochs

        train_acc, train_obj, train_top5 = train_evaluation_phase(
            args,
            train_queue,
            model,
            criterion,
            optimizer,
            scheduler,
            rank
        )
        train_acc_tensor = torch.tensor(train_acc).cuda(rank)
        train_obj_tensor = torch.tensor(train_obj).cuda(rank)
        train_top5_tensor = torch.tensor(train_top5).cuda(rank)
        dist.reduce(train_acc_tensor, dst=0)
        dist.reduce(train_obj_tensor, dst=0)
        dist.reduce(train_top5_tensor, dst=0)

        if rank == 0:
            train_acc_mean = train_acc_tensor / world_size
            train_obj_mean = train_obj_tensor / world_size
            train_top5_mean = train_top5_tensor / world_size

            logging.info(f"| train_acc: {train_acc_mean} |")
            # Log values
            writer.add_scalar("Loss/train", train_obj_mean.item(), epoch)
            writer.add_scalar("Top1/train", train_acc_mean.item(), epoch)
            writer.add_scalar("Top5/train", train_top5_mean.item(), epoch)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        #    dist.barrier()
        #else:
        #    dist.barrier()

        valid_acc, valid_obj, valid_top5 = train_utils.infer(
            valid_queue,
            model,
            criterion,
            report_freq=args.run.report_freq
        )
        valid_acc_tensor = torch.tensor(valid_acc).cuda(rank)
        valid_obj_tensor = torch.tensor(valid_obj).cuda(rank)
        valid_top5_tensor = torch.tensor(valid_top5).cuda(rank)

        dist.reduce(valid_acc_tensor, dst=0)
        dist.reduce(valid_obj_tensor, dst=0)
        dist.reduce(valid_top5_tensor, dst=0)
        
        # memory stats
        mem_peak_allocated_MB = torch.cuda.max_memory_allocated() / 1e6
        mem_peak_allocated_MB = torch.tensor(mem_peak_allocated_MB)
        mem_peak_allocated_MB = mem_peak_allocated_MB.cuda(rank)
        #mem_peak_allocated_MB = torch.tensor(torch.cuda.max_memory_allocated() / 1e6).cuda(rank)
        mem_peak_reserved_MB = torch.tensor(torch.cuda.max_memory_reserved() / 1e6).cuda(rank)

        logging.info(f"Max allocated GPU {rank}: {mem_peak_allocated_MB}")
        logging.info(f"MAx reserved GPU {rank}: {mem_peak_reserved_MB}")

        dist.reduce(mem_peak_allocated_MB, dst=0)
        dist.reduce(mem_peak_reserved_MB, dst=0)

        if rank == 0:
            valid_acc_mean = valid_acc_tensor / world_size
            valid_obj_mean = valid_obj_tensor / world_size
            valid_top5_mean = valid_top5_tensor / world_size
            logging.info(f"reduced mem_peak_allocated_MB = {mem_peak_allocated_MB}")
            logging.info(f"reduced mem_peak_reserved_MB = {mem_peak_reserved_MB}")
            mem_peak_allocated_MB_mean = mem_peak_allocated_MB / world_size
            logging.info(f"mem_peak_allocated_MB_mean = {mem_peak_allocated_MB_mean}")
            mem_peak_reserved_MB_mean = mem_peak_reserved_MB / world_size
            logging.info(f"mem_peak_reserved_MB_mean = {mem_peak_reserved_MB_mean}")
            global_peak_mem_allocated_MB = max(global_peak_mem_allocated_MB, mem_peak_allocated_MB_mean.item())
            global_peak_mem_reserved_MB = max(global_peak_mem_reserved_MB, mem_peak_reserved_MB_mean.item())
            logging.info(f"Global max allocated: {global_peak_mem_allocated_MB}")
            logging.info(f"Global max reserved: {global_peak_mem_reserved_MB}")
            logging.info(f"| valid_acc: {valid_acc_mean} |")
            
            writer.add_scalar("Loss/valid", valid_obj_mean.item(), epoch)
            writer.add_scalar("Top1/valid", valid_acc_mean.item(), epoch)
            writer.add_scalar("Top5/valid", valid_top5_mean.item(), epoch)

            writer.add_scalar("Mem/peak_allocated_MB", global_peak_mem_allocated_MB, epoch)
            writer.add_scalar("Mem/peak_reserved_MB", global_peak_mem_reserved_MB, epoch)

            # Use validation accuracy to determine if we have obtained new best weights
            if valid_acc_mean > best_observed['valid']:
                best_observed['train'] = train_acc_mean.item()
                best_observed['valid'] = valid_acc_mean.item()
                best_observed['epoch'] = epoch
                best_observed['runtime'] = timer() - train_start_time + previous_runtime

                # best_eval=True indicates that we want to separately save this checkpoint, so that at the end, we can load
                # the weights with the best performance according to validation data
                train_utils.save(
                    checkpoint_dir,
                    epoch+1,
                    rng_seed,
                    model,
                    optimizer,
                    runtime=(timer() - train_start_time + previous_runtime),
                    best_observed=best_observed,
                    best_eval=True,
                    multi_process=True,
                    max_mem_allocated_MB=global_peak_mem_allocated_MB,
                    max_mem_reserved_MB=global_peak_mem_reserved_MB
                )

            # Save checkpoint for current epoch
            train_utils.save(
                checkpoint_dir,
                epoch+1,
                rng_seed,
                model,
                optimizer,
                runtime=(timer() - train_start_time + previous_runtime),
                best_observed=best_observed,
                multi_process=True,
                max_mem_allocated_MB=global_peak_mem_allocated_MB,
                max_mem_reserved_MB=global_peak_mem_reserved_MB
            )
        #    dist.barrier()
        #else:
        #    dist.barrier()
        if args.train.scheduler != "cosine_mgpu":
            scheduler.step()

    # Training finished
    if rank == 0:
        train_end_time = timer()
        overall_runtime = train_end_time - train_start_time + previous_runtime
        logging.info(f"Training finished after {timedelta(seconds=overall_runtime)} hh:mm:ss.")

        logging.info(
            (
                f"Best weights according to validation accuracy found in epoch {best_observed['epoch']} after "
                f"{timedelta(seconds=best_observed['runtime'])} hh:mm:ss."
            )
        )
        logging.info(f"Train accuracy of best weights: {best_observed['train']} %")
        logging.info(f"Validation accuracy of best weights: {best_observed['valid']} %")
        logging.info(f"\nCheckpoint of best weights can be found in: {os.path.join(checkpoint_dir, 'model_best.ckpt')}")
        result_dict = {
            'checkpoint_path': os.path.join(checkpoint_dir, 'model_best.ckpt'),
            'runtime_best_observed': best_observed['runtime'],
            'train_acc_best_observed': best_observed['train'],
            'valid_acc_best_observed': best_observed['valid'],
            'overall_runtime': overall_runtime,
            'max_mem_allocated_MB': global_peak_mem_allocated_MB,
            'max_mem_reserved_MB': global_peak_mem_reserved_MB,
            'total_params': total_params
        }
        result_queue.put(result_dict)
        logging.info(f"Results of evaluation put into queue.")
        # before return, remove logging filehandler of current logfile, so that the following logs aren't written in the current log
        logging.getLogger().removeHandler(logging.getLogger().handlers[-1])

    # if rank == 0:
    #     logging.info("Entering barrier before destroying process group")
    print(f'Rank {rank} before destroying process group', flush=True)
    #dist.barrier()
    # if rank == 0:
    #     logging.info("Exited barrier")
    #     logging.getLogger().removeHandler(logging.getLogger().handlers[-1]) # when removing this line, uncomment the logger line above!
    dist.destroy_process_group()
    

def search_phase(args, base_dir, run_id):
    """Performs NAS.
    Code is mostly copied from train_search.py but modified to only work with GAEA PC-DARTS.
    Best genotype is selected according to training accuracy for single-level search and according to validation
        accuracy for bi-level search.

    Args:
        args (OmegaConf): Arguments.
        base_dir (str): Path to the base directory that the search phase should work in.
        run_id (str): Identifier to distinguish multiple runs.

    Returns:
        Genotype: Best found genotype.
        float: Runtime in seconds after which the best genotype was found.
        float: Train accuracy of the best found genotype.
        float: Validation accuracy of the best found genotype.
        float: Overall runtime of the search phase in seconds.
        float: Maximum memory allocated in MB.
        float: Maximum memory reserved in MB.
    """
    # Create folder structure
    try:
        run_identifier = f"{run_id}_{str(args.train[run_id])}"
    except KeyError:
        run_identifier = f"invalid_run_id_{run_id}"

    log_dir = os.path.join(base_dir, "logs")
    summary_dir = os.path.join(base_dir, "summary")
    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    genotype_dir = os.path.join(base_dir, "genotypes")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", f"checkpoint_{run_identifier}")
    for directory in [log_dir, summary_dir, tensorboard_dir, genotype_dir, checkpoint_dir]:
        os.makedirs(directory, exist_ok=True)

    # Log file for the current search phase
    logfile = os.path.join(log_dir, f"log_{run_identifier}.txt")
    train_utils.set_up_logging(logfile)

    logging.info(f"Hyperparameters: \n{args.pretty()}")

    # Setup SummaryWriters
    summary_writer_dir = os.path.join(summary_dir, run_identifier)
    tensorboard_writer_dir = os.path.join(tensorboard_dir, run_identifier)
    writer = SummaryWriter(summary_writer_dir)
    # own writer that I use to keep track of interesting variables
    own_writer = SummaryWriter(tensorboard_writer_dir)

    #if not torch.cuda.is_available():
    #    logging.error("No GPU device available")
    #    sys.exit(-1)
    #torch.cuda.set_device(args.run.gpu)
    #torch.backends.cudnn.benchmark=True
    if not torch.cuda.is_available():
        raise Exception("No GPU device available")
    torch.cuda.set_device(args.run.gpu)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark=True

    current_device = torch.cuda.current_device()
    logging.info(f"Current cuda device: {current_device} - {torch.cuda.get_device_name(current_device)}")

    # reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Set random seeds for random, numpy, torch and cuda
    rng_seed = train_utils.RNGSeed(args.run.seed)

    if args.train.smooth_cross_entropy:
        criterion = train_utils.cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    # old code that directly obtains data queues
    # if single-level, train_2_queue points to the training data. During bi-level search, train_2_queue will be None and we'll use valid_queue for search
    #num_classes, (train_queue, train_2_queue), valid_queue, test_queue, (number_train, number_valid, number_test) = train_utils.create_cifar10_data_queues_own(args)

    if args.run.dataset != "cifar10":
        raise ValueError(f"Only cifar10 dataset is supported, got {args.run.dataset}")

    # Get datasets; valid_dataset is the same as train_dataset besides different data transformations
    train_dataset, valid_dataset, test_dataset = train_utils.get_cifar10_data_sets(args)
    num_test = len(test_dataset)
    del test_dataset    # don't want to have anything to do with test data here apart from getting the number of test samples
    num_train_overall = len(train_dataset)
    train_indices_overall = np.arange(num_train_overall)
    # split train and validation data
    if args.search.single_level:
        # validation data is not used during search
        train_valid_split = int(np.floor(num_train_overall * args.train.train_portion_single_level))
        assert train_valid_split % 2 == 0, f"Train data must be splittable into two subsets of the same size, but is of size {len(train_valid_split)}"
        train_end = int(np.floor(train_valid_split / 2))    # point at which training data is split into data for weight updates and data for architecture updates
    else:
        # validation data is used to update architectural weights
        valid_dataset = deepcopy(train_dataset) # should have same data transformation as train data
        train_valid_split = int(np.floor(num_train_overall * args.train.train_portion_bi_level))
        assert len(train_indices_overall[:train_valid_split]) == len(train_indices_overall[train_valid_split:]), "Train and validation dataset must have same size"
        train_end = train_valid_split
    
    num_train = train_valid_split
    num_valid = num_train_overall - num_train
    num_classes = 10

    # Random samplers for data queues
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices_overall[:train_end])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices_overall[train_valid_split:])
    if args.search.single_level:
        train_2_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices_overall[train_end:train_valid_split])  # used for architecture updates

    # train queue will change during single-level search if single_level_shuffle is true
    train_queue = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.run.n_threads_data
    )

    if args.search.single_level:
        # train_2_queue will change during single-level search if single_level_shuffle is true
        train_2_queue = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.train.batch_size,
            sampler=train_2_sampler,
            pin_memory=True,
            num_workers=args.run.n_threads_data
        )

    # validation queue stays constant over whole search phase
    valid_queue = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.train.batch_size,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=args.run.n_threads_data
    )

    logging.info(f"Dataset: {args.run.dataset}")
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Number of training images: {num_train}")
    if args.search.single_level:
        logging.info(f"Number of validation images (unused during search): {num_valid}")
    else:
        logging.info(f"Number of validation images (used during search): {num_valid}")
    logging.info(f"Number of test images (unused during search): {num_test}")

    # Create model
    model = PCDARTSNetwork(
        args.train.init_channels,
        num_classes,
        args.search.nodes,
        args.train.layers,
        criterion,
        **{
            "auxiliary": args.train.auxiliary,
            "search_space_name": args.search.search_space,
            "exclude_zero": args.search.exclude_zero,
            "track_running_stats": args.search.track_running_stats,
        }
    )
    model = model.cuda()
    logging.info(f"Model parameter size: {train_utils.count_parameters_in_MB(model)} MB")

    optimizer, scheduler = train_utils.setup_optimizer(model, args)

    # Create architect
    architect = Architect(model, args, writer)

    # Try to load previous checkpoint
    try:
        (
            start_epochs, 
            history, 
            previous_runtime, 
            best_observed,
            global_peak_mem_allocated_MB,
            global_peak_mem_reserved_MB 
        ) = train_utils.load(
            checkpoint_dir,
            rng_seed,
            model,
            optimizer,
            architect,
            args.run.s3_bucket
        )
        scheduler.last_epoch = ((start_epochs -1) * len(train_queue)) if args.train.scheduler == "cosine_mgpu" else (start_epochs - 1)
        if best_observed is None:
            best_observed = {
                "train": 0.0,           # for single-level search, used to keep track of best genotype
                "valid": 0.0,           # for bi-level search, used to keep track of best genotype
                "epoch": 0,             # epoch the best accuracy was observed
                "genotype_raw": None,   # best genotype
                "genotype_dict": None,  # best genotype stored as dict (for serialization)
                "runtime": 0.0          # runtime after which the best genotype was found
            }
        logging.info(
            (
                f"Resumed training from a previous checkpoint which was already trained for {start_epochs} epochs and "
                f"already ran for {timedelta(seconds=previous_runtime)} hh:mm:ss."
            )
        )
        logging.info("This is included in the final runtime report.")
    except Exception as e:
        logging.info(e)
        start_epochs = 0
        previous_runtime = 0
        global_peak_mem_allocated_MB = 0.
        global_peak_mem_reserved_MB = 0.

        best_observed = {
            "train": 0.0,           # for single-level search, used to keep track of best genotype
            "valid": 0.0,           # for bi-level search, used to keep track of best genotype
            "epoch": 0,             # epoch the best accuracy was observed
            "genotype_raw": None,   # best genotype
            "genotype_dict": None,  # best genotype stored as dict (for serialization)
            "runtime": 0.0          # runtime after which the best genotype was found
        }
    
    logging.info("Search phase started")

    train_start_time = timer()
    
    overall_visualization_time = 0      # don't count visualization into runtime

    # Train loop
    for epoch in range(start_epochs, args.run.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info(f"| Epoch: {epoch:3d} / {args.run.epochs} | lr: {lr} |")

        model.drop_path_prob = args.train.drop_path_prob * epoch / args.run.epochs

        # during single-level search, shuffle training samples such that they can appear in both training queues (considering all epochs)
        if args.search.single_level and args.search.single_level_shuffle:
            train_indices = np.arange(num_train)    # num_train == train_valid_split
            np.random.shuffle(train_indices)
            split = int(np.floor(0.5 * num_train))

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices[:split])
            train_2_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices[split:])

            train_queue = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.train.batch_size,
                sampler=train_sampler,
                pin_memory=True,
                num_workers=args.run.n_threads_data
            )

            train_2_queue = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.train.batch_size,
                sampler=train_2_sampler,
                pin_memory=True,
                num_workers=args.run.n_threads_data
            )

        # training returns top1, loss and top5
        train_acc, train_obj, train_top5 = train_search_phase(
            args, train_queue, train_2_queue if args.search.single_level else valid_queue,
            model, architect, criterion, optimizer, scheduler, lr,
        )
        architect.baseline = train_obj
        architect.update_history()
        architect.log_vars(epoch, writer)

        if "update_lr_state" in dir(scheduler):
            scheduler.update_lr_state(train_obj)

        logging.info(f"| train_acc: {train_acc} |")

        # History tracking
        for vs in [("alphas", architect.alphas), ("edges", architect.edges)]:
            for ct in vs[1]:
                v = vs[1][ct]
                logging.info("{}-{}".format(vs[0], ct))
                logging.info(v)
        # Calling genotypes sets alphas to best arch for EGDAS and MEGDAS
        # so calling here before infer.
        genotype = architect.genotype()
        logging.info("genotype = %s", genotype)

        # log epoch values to tensorboard
        own_writer.add_scalar('Loss/train', train_obj, epoch)
        own_writer.add_scalar('Top1/train', train_acc, epoch)
        own_writer.add_scalar('Top5/train', train_top5, epoch)
        own_writer.add_scalar('lr', lr, epoch)

        # visualize Genotype
        start_visualization = timer()
        genotype_graph_normal = visualize.plot(genotype.normal, "", return_type="graph", output_format='png')
        binary_normal = genotype_graph_normal.pipe()
        stream_normal = io.BytesIO(binary_normal)
        graph_normal = np.array(PIL.Image.open(stream_normal).convert("RGB"))
        own_writer.add_image("Normal_Cell", graph_normal, epoch, dataformats="HWC")
        #del genotype_graph_normal
        #del binary_normal
        #del stream_normal
        #del graph_normal

        genotype_graph_reduce = visualize.plot(genotype.reduce, "", return_type="graph", output_format='png')
        binary_reduce = genotype_graph_reduce.pipe()
        stream_reduce = io.BytesIO(binary_reduce)
        graph_reduce = np.array(PIL.Image.open(stream_reduce).convert("RGB"))
        own_writer.add_image("Reduce_Cell", graph_reduce, epoch, dataformats="HWC")
        #del genotype_graph_reduce
        #del binary_reduce
        #del stream_reduce
        #del graph_reduce
        end_visualization = timer()
        overall_visualization_time += (end_visualization - start_visualization)

        # log validation metrics, but don't utilize them for decisions during single-level search 
        valid_acc, valid_obj, valid_top5 = train_utils.infer(
            valid_queue,
            model,
            criterion,
            report_freq=args.run.report_freq,
            discrete=args.search.discrete,
        )
        own_writer.add_scalar('Loss/valid', valid_obj, epoch)
        own_writer.add_scalar('Top1/valid', valid_acc, epoch)
        own_writer.add_scalar('Top5/valid', valid_top5, epoch)
        logging.info(f"| valid_acc: {valid_acc} |")
        # memory stats
        global_peak_mem_allocated_MB = max(global_peak_mem_allocated_MB, torch.cuda.max_memory_allocated() / 1e6)
        global_peak_mem_reserved_MB = max(global_peak_mem_reserved_MB, torch.cuda.max_memory_reserved() / 1e6)
        own_writer.add_scalar("Mem/peak_allocated_MB", global_peak_mem_allocated_MB, epoch)
        own_writer.add_scalar("Mem/peak_reserved_MB", global_peak_mem_reserved_MB, epoch)

        if (args.search.single_level and train_acc > best_observed['train']) or (not args.search.single_level and valid_acc > best_observed['valid']):
                best_observed['train'] = train_acc
                best_observed['valid'] = valid_acc
                best_observed['epoch'] = epoch
                best_observed['genotype_raw'] = genotype
                best_observed['genotype_dict'] = train_utils.genotype_to_dict(genotype)
                best_observed['runtime'] = timer() - train_start_time - overall_visualization_time + previous_runtime

        # Save checkpoint of this epoch
        train_utils.save(
            checkpoint_dir,
            epoch + 1,
            rng_seed,
            model,
            optimizer,
            architect,
            save_history=True,
            s3_bucket=args.run.s3_bucket,
            runtime=(timer()-train_start_time - overall_visualization_time + previous_runtime),
            best_observed=best_observed,
            max_mem_allocated_MB=global_peak_mem_allocated_MB,
            max_mem_reserved_MB=global_peak_mem_reserved_MB
        )

        if args.train.scheduler != "cosine_mgpu":
            scheduler.step()

    train_end_time = timer()
    overall_runtime = train_end_time - train_start_time - overall_visualization_time + previous_runtime
    logging.info(f"Visualization of cells during search took a total of {timedelta(seconds=overall_visualization_time)} (hh:mm:ss).")
    logging.info(f"This time is not included in the runtime given below.\n")
    logging.info(f"Training finished after {timedelta(seconds=overall_runtime)} hh:mm:ss.")

    if args.search.single_level:
        logging.info(
            (
                f"\nBest genotype according to training accuracy found in epoch {best_observed['epoch']} after "
                f"{timedelta(seconds=best_observed['runtime'])} hh:mm:ss"
            )
        )
    else:
        logging.info(
            (
                f"\nBest genotype according to validation accuracy found in epoch {best_observed['epoch']} after "
                f"{timedelta(seconds=best_observed['runtime'])} hh:mm:ss"
            )
        )
    logging.info(f"Train accuracy: {best_observed['train']} %")
    logging.info(f"Validation accuracy: {best_observed['valid']} %")
    logging.info(f"Genotype: {best_observed['genotype_raw']}")

    # dump best genotype to json file, so that we can load it during evaluation phase
    genotype_file_path = os.path.join(genotype_dir, f"genotype_{run_identifier}.json")
    with open(genotype_file_path, 'w') as genotype_file:
        json.dump(best_observed['genotype_dict'], genotype_file, indent=4)

    logging.info(f"Search finished. Dumped best genotype into {genotype_file_path}")

    # before return, remove logging filehandler of current logfile, so that the following logs aren't written in the current log
    logging.getLogger().removeHandler(logging.getLogger().handlers[-1])
    return (
        best_observed['genotype_raw'],
        best_observed['runtime'],
        best_observed['train'],
        best_observed['valid'],
        overall_runtime,
        global_peak_mem_allocated_MB,
        global_peak_mem_reserved_MB
    )
    

def train_search_phase(
    args,
    train_queue,
    valid_queue,
    model,
    architect,
    criterion,
    optimizer,
    scheduler,
    lr,
    random_arch=False
):
    """Train routine for architecture search phase.

    Args:
        args (OmegaConf): Arguments
        train_queue (torch.utils.DataLoader): Training dataset.
        valid_queue (torch.utils.DataLoader): Validation dataset.
            When utilizing single-level search, this is supposed to be a 
                DataLoader that also points to the training data and !!NOT!! the actual validation data.
            When utilizing bi-level search, this is supposed to be a 
                DataLoader that points to the validation data.
        model (nn.Module): The model that should be trained.
        architect (Architect): Architect that should be used to update architecture parameters.
        criterion (callable): Loss that should be utilized for weight updates.
        optimizer: The optimizer that should be utilized for weight updates.
        scheduler: The learning rate scheduler. Needed in case of cosine_mgpu.
        lr (float): Current learning rate.
        random_arch (bool): Should a random architecture be returned.
            TODO: Might not work at the moment.

    Returns:
        float: Top1 accuracy
        float: Loss
        float: Top5 accuracy
    """
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()

    for step, datapoint in enumerate(train_queue):
        if "nas-bench-201" in args.search.search_space:
            raise ValueError("NAS-Bench-201 is not supported")

        input, target = datapoint
        input_search, target_search = next(iter(valid_queue))

        batch_size = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        # set the model in train mode (important for layers like dropout and batch normalization)
        model.train()

        if not random_arch:
            architect.step(
                input,
                target,
                input_search,
                target_search,
                **{
                    "eta": lr,
                    "network_optimizer": optimizer,
                    "unrolled": args.search.unrolled,
                    "update_weights": True,
                }
            )

        optimizer.zero_grad()
        architect.zero_arch_var_grad()
        architect.set_model_alphas()
        architect.set_model_edge_weights()

        logits, logits_aux = model(input, discrete=args.search.discrete)
        loss = criterion(logits, target)
        if args.train.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.train.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.train.grad_clip)
        optimizer.step()
        if args.train.scheduler == "cosine_mgpu":
            scheduler.step()

        prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        if step % args.run.report_freq == 0:
            logging.info(f"| Train | Batch: {step:3d} | Loss: {objs.avg:e} | Top1: {top1.avg} | Top5: {top5.avg} |")

    return top1.avg, objs.avg, top5.avg


def train_evaluation_phase(
    args,
    train_queue,
    model,
    criterion,
    optimizer,
    scheduler,
    rank
):
    """Train routine for architecture evaluation phase.

    Args:
        args (OmegaConf): Arguments
        train_queue (torch.utils.DataLoader): Training dataset.
        model (torch.nn.Module): The model that should be trained.
        criterion (callable): Loss that should be used for weight updates.
        optimizer: The optimizer that should be used for weight updates.
        scheduler: The learning rate scheduler. Needed in case of cosine_mgpu.
        rank (int): Rank of the current process

    Returns:
        float: Training accuracy.
        float: Training loss.
        float: Training top5 accuracy.
    """
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()

    # set model to training mode
    model.train()

    for step, (data, target) in enumerate(train_queue):
        data = Variable(data, requires_grad=False).cuda(non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(data)
        loss = criterion(logits, target)

        if args.train.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.train.auxiliary_weight * loss_aux
        
        loss.backward()         # calculates dloss / dx for every parameter x
        nn.utils.clip_grad_norm_(model.parameters(), args.train.grad_clip)
        optimizer.step()        # performs gradient update for every x
        if args.train.scheduler == "cosine_mgpu":
            scheduler.step()

        prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5))
        batch_size = data.size(0)
        objs.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        if step % args.run.report_freq == 0 and rank == 0:
            logging.info(f"| Batch: {step:3d} | Loss: {objs.avg:5f} | Top1: {top1.avg:3f} | Top5: {top5.avg:3f} |")

    return top1.avg, objs.avg, top5.avg


if __name__ == '__main__':
    main()
