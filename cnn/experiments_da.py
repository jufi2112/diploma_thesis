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
from collections import namedtuple

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from datetime import timedelta
from timeit import default_timer as timer

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
    log = os.path.join(working_directory, "overall_log.txt")
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
            best_init_channels = eval_history.keys()[0]

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
            

            best_init_channels = search_history.keys()[0]
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
        raise NotImplementedError('Gaussian process is currently not implemented.')
    else:
        raise ValueError("Unrecognized method.")



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
    cwd = os.getcwd()
    log = os.path.join(cwd, "log_grid_search.txt")
    train_utils.set_up_logging(log)

    logging.info(f"Hyperparameters: \n{args.pretty()}")

    # overall directory (cwd) is set up by hydra
    # base directories for search and evaluation phases
    base_dir_search = os.path.join(cwd, "search_phase_seed_" + str(args.run_search_phase.seed))
    base_dir_eval = os.path.join(cwd, "evaluation_phase_seed_" + str(args.run_eval_phase.seed))

    logging.info(f"Starting grid search with mode: {args.method.mode}")

    overall_runtime_search_phase = 0.0  # measures the time spend search
    overall_runtime_eval_phase = 0.0    # measures the time spend evaluating

    # can have mode 'search_only'   - perform only search phase
    #               'evaluate_only' - perform only evaluation phase
    #               'sequential'    - perform search phase followed by evaluation phase
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
                    ) = search_phase(args, base_dir_search)
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
                        current_init_channels,
                        search_history[current_init_channels]['best_genotype']
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
        except:
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
        for step, current_init_channels in enumerate(search_history.keys()):
            logging.info(f"| Step {(step+1)} / {len(search_history.keys())} |")
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
                    current_init_channels,
                    search_history[current_init_channels]['best_genotype']
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


def evaluation_phase(args, base_dir, genotype_init_channels, genotype_to_evaluate):
    """Fully trains a provided genotype
    Code is mostly copied from train_final.py but modified to work for my experiments.
    Best weights are selected according to validation accuracy.

    Args:
        args (OmegaConf): Arguments
        base_dir (str): Path to the base directory the evaluation phase should work in.
        genotype_init_channels (int): Initial number of channels the genotype was searched with.
            Gets used as ID for different evaluation runs.
            This does NOT influence init_channels for evaluation phase, this is controlled via args.train.init_channels!
        genotype_to_evaluate (Genotype or str): Genotype that should be evaluated.
            A string is interpreted as path to a json file that contains the genotype that should be evaluated.

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
    # Create folder structure
    #base_dir = os.path.join(os.getcwd(), "evaluation_phase_seed_" + str(args.run.seed))
    log_dir = os.path.join(base_dir, "logs")
    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", "checkpoint_init_channels_" + str(genotype_init_channels))

    # Log file for the current evaluation phase
    logfile = os.path.join(log_dir, "log_init_channels_" + str(genotype_init_channels) + ".txt")
    train_utils.set_up_logging(logfile)

    logging.info(f"Hyperparameters: \n{args.pretty()}")

    # Tensorboard SummaryWriter setup
    tensorboard_writer_dir = os.path.join(tensorboard_dir, "init_channels_" + str(genotype_init_channels))
    writer = SummaryWriter(tensorboard_writer_dir)

    if not torch.cuda.is_available():
        logging.error("No GPU device available!")
        sys.exit(-1)
    torch.cuda.set_device(args.run.gpu)

    # reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Set random seeds for random, numpy, torch and cuda
    rng_seed = train_utils.RNGSeed(args.run.seed)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # Load datasets
    num_classes, (train_queue, _), valid_queue, test_queue, (num_train, num_valid, num_test) = train_utils.create_cifar10_data_queues_own(
        args, eval_split=True
    )

    logging.info(f"Dataset: {args.run.dataset}")
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Number of training images: {number_train}")
    logging.info(f"Number of validation images: {number_valid}")
    logging.info(f"Number of test images: {number_test}")

    # Load genotype
    if type(genotype_to_evaluate) == str:
        try:
            with open(genotype_to_evaluate) as genotype_file:
                genotype_dict = json.load(genotype_file)
            genotype_to_evaluate = train_utils.dict_to_genotype(genotype_dict)
        except Exception as e:
            logging.error(f"Error while trying to load genotype from the provided file: \n{e}")
            return # TODO: how should errors be returned?

    # Visualize genotype
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
    model = model.cuda()

    logging.info(f"Size of model parameters: {train_utils.count_parameters_in_MB(model)} MB")
    total_params = sum(x.data.nelement() for x in model.parameters())
    logging.info(f"Total parameters of model: {total_params}")

    optimizer, scheduler = train_utils.setup_optimizer(model, args)

    # Check if we've already trained
    try:
        start_epochs, _, previous_runtime, best_observed = train_utils.load(
            checkpoint_dir, rng_seed, model, optimizer, s3_bucket=None
        )
        if best_observed is None:
            best_observed = {
                "train": 0.0,           # train accuracy of best epoch
                "valid": 0.0,           # validation accuracy of best epoch
                "epoch": 0,             # epoch the best accuracy was observed
                "genotype_raw": None,   # not used, but needs to be in the dict for compatibility with search phase
                "genotype_dict": None,  # not used, but needs to be in the dict for compatibility with search phase
                "runtime": 0.0          # runtime after which the best epoch was observed
            }
        scheduler.last_epoch = start_epochs - 1
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

        best_observed = {
            "train": 0.0,           # train accuracy of best epoch
            "valid": 0.0,           # validation accuracy of best epoch
            "epoch": 0,             # epoch the best accuracy was observed
            "genotype_raw": None,   # not used, but needs to be in the dict for compatibility with search phase
            "genotype_dict": None,  # not used, but needs to be in the dict for compatibility with search phase
            "runtime": 0.0          # runtime after which the best epoch was observed
        }

    logging.info(f"Evaluation phase started for genotype: \n{genotype_to_evaluate}")
    logging.info(f"The genotype was searched with init_channels = {genotype_init_channels}")
    train_start_time = timer()

    # Train loop
    for epoch in range(start_epochs, args.run.epochs):
        logging.info(f"| Epoch: {epoch:4d}/{args.run.epochs} | lr: {scheduler.get_lr()[0]} |")
        model.drop_path_prob = args.train.drop_path_prob * epoch / args.run.epochs

        train_acc, train_obj, train_top5 = train_evaluation_phase(
            args,
            train_queue,
            model,
            criterion,
            optimizer
        )
        logging.info(f"| train_acc: {train_acc} |")

        valid_acc, valid_obj, valid_top5 = train_utils.infer(
            valid_queue,
            model,
            criterion,
            report_freq=args.run.report_freq
        )
        logging.info(f"| valid_acc: {valid_acc} |")

        # Log values
        writer.add_scalar("Loss/train", train_obj, epoch)
        writer.add_scalar("Top1/train", train_acc, epoch)
        writer.add_scalar("Top5/train", train_top5, epoch)
        writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        writer.add_scalar("Loss/valid", valid_obj, epoch)
        writer.add_scalar("Top1/valid", valid_acc, epoch)
        writer.add_scalar("Top5/valid", valid_top5, epoch)
        # memory stats
        mem_peak_allocated_MB = torch.cuda.max_memory_allocated() / 1e6
        mem_peak_reserved_MB = torch.cuda.max_memory_reserved() / 1e6
        writer.add_scalar("Mem/peak_allocated_MB", mem_peak_allocated_MB, epoch)
        writer.add_scalar("Mem/peak_reserved_MB", mem_peak_reserved_MB, epoch)

        # Use validation accuracy to determine if we have obtained new best weights
        if valid_acc > best_observed['valid']:
            best_observed['train'] = train_acc
            best_observed['valid'] = valid_acc
            best_observed['epoch'] = epoch
            best_observed['runtime'] = timer() - train_start_time + previous_runtime

            # best_eval=True indicates that we want to separately save this checkpoint, so that at the end, we can load
            # the weights with the best performance according to validation data
            train_utils.save(
                checkpoint_dir,
                epochs+1,
                rng_seed,
                model,
                optimizer,
                runtime=(timer() - train_start_time + previous_runtime),
                best_observed=best_observed,
                best_eval=True
            )

        # Save checkpoint for current epoch
        train_utils.save(
            checkpoint_dir,
            epochs+1,
            rng_seed,
            model,
            optimizer,
            runtime=(timer() - training_start_time + previous_runtime),
            best_observed=best_observed
        )

        scheduler.step()

    train_end_time = timer()
    overall_runtime = train_end_time - train_start_time + previous_runtime
    logging.info(f"\nTraining finished after {timedelta(seconds=overall_runtime)} hh:mm:ss.")

    logging.info(
        (
            f"Best weights according to validation accuracy found in epoch {best_observed['epoch']} after "
            f"{timedelta(seconds=best_observed['runtime'])} hh:mm:ss."
        )
    )
    logging.info(f"Train accuracy of best weights: {best_observed['train']} %")
    logging.info(f"Validation accuracy of best weights: {best_observed['valid']} %")
    logging.info(f"\nCheckpoint of best weights can be found in: {os.path.join(checkpoint_dir, 'model_best.ckpt')}")
        
    # before return, remove logging filehandler of current logfile, so that the following logs aren't written in the current log
    logging.getLogger().removeHandler(logging.getLogger().handlers[-1])
    return (
        os.path.join(checkpoint_dir, 'model_best.ckpt'),
        best_observed['runtime'],
        best_observed['train'],
        best_observed['valid'],
        overall_runtime,
        torch.cuda.max_memory_allocated() / 1e6,
        torch.cuda.max_memory_reserved() / 1e6,
        total_params
    )
    
    
    
        


def search_phase(args, base_dir):
    """Performs NAS
    Code is mostly copied from train_search.py but modified to only work with GAEA PC-DARTS.
    Best genotype is selected according to training accuracy for single-level search and according to validation
        accuracy for bi-level search.

    Args:
        args (OmegaConf): Arguments.
        base_dir (str): Path to the base directory that the search phase should work in.

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
    #base_dir = os.path.join(os.getcwd(), "search_phase_seed_" + str(args.run.seed))
    log_dir = os.path.join(base_dir, "logs")
    summary_dir = os.path.join(base_dir, "summary")
    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    genotype_dir = os.path.join(base_dir, "genotypes")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", "checkpoint_init_channels_" + str(args.train.init_channels))
    for directory in [log_dir, summary_dir, tensorboard_dir, genotype_dir, checkpoint_dir]:
        os.makedirs(directory, exist_ok=True)

    # Log file for the current search phase
    logfile = os.path.join(log_dir, "log_init_channels_" + str(args.train.init_channels) + ".txt")
    train_utils.set_up_logging(logfile)

    logging.info(f"Hyperparameters: \n{args.pretty()}")

    # Setup SummaryWriters
    summary_writer_dir = os.path.join(summary_dir, "init_channels_" + str(args.train.init_channels))
    tensorboard_writer_dir = os.path.join(tensorboard_dir, "init_channels_" + str(args.train.init_channels))
    writer = SummaryWriter(summary_writer_dir)
    # own writer that I use to keep track of interesting variables
    own_writer = SummaryWriter(tensorboard_writer_dir)

    if not torch.cuda.is_available():
        logging.error("No GPU device available")
        sys.exit(-1)
    torch.cuda.set_device(args.run.gpu)

    # reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Set random seeds for random, numpy, torch and cuda
    rng_seed = train_utils.RNGSeed(args.run.seed)

    if args.train.smooth_cross_entropy:
        criterion = train_utils.cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    # if single-level, train_2_queue points to the training data. During bi-level search, train_2_queue will be None and we'll use valid_queue for search
    num_classes, (train_queue, train_2_queue), valid_queue, test_queue, (number_train, number_valid, number_test) = train_utils.create_cifar10_data_queues_own(args)

    logging.info(f"Dataset: {args.run.dataset}")
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Number of training images: {number_train}")
    if args.search.single_level:
        logging.info(f"Number of validation images (unused during search): {number_valid}")
    else:
        logging.info(f"Number of validation images (used during search): {number_valid}")
    logging.info(f"Number of test images (unused during search): {number_test}")

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
        start_epochs, history, previous_runtime, best_observed = train_utils.load(
            checkpoint_dir,
            rng_seed,
            model,
            optimizer,
            architect,
            args.run.s3_bucket
        )
        scheduler.last_epoch = start_epochs - 1
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
        lr = scheduler.get_lr()[0]
        logging.info(f"| Epoch: {epoch:3d} / {args.run.epochs} | lr: {lr} |")

        model.drop_path_prob = args.train.drop_path_prob * epoch / args.run.epochs

        # training returns top1, loss and top5
        train_acc, train_obj, train_top5 = train_search_phase(
            args, train_queue, valid_queue if train_2_queue == None else train_2_queue, # valid_queue for bi-level search, train_2_queue for single-level search
            model, architect, criterion, optimizer, lr,
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
        mem_peak_allocated_MB = torch.cuda.max_memory_allocated() / 1e6
        mem_peak_reserved_MB = torch.cuda.max_memory_reserved() / 1e6
        writer.add_scalar("Mem/peak_allocated_MB", mem_peak_allocated_MB, epoch)
        writer.add_scalar("Mem/peak_reserved_MB", mem_peak_reserved_MB, epoch)

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
            best_observed=best_observed
        )

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
    genotype_file_path = os.path.join(genotype_dir, "genotype_init_channels_" + str(args.train.init_channels) + ".json")
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
        torch.cuda.max_memory_allocated() / 1e6,
        torch.cuda.max_memory_reserved() / 1e6
    )
    
    
        


def train_search_phase(
    args,
    train_queue,
    valid_queue,
    model,
    architect,
    criterion,
    optimizer,
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
    optimizer
):
    """Train routine for architecture evaluation phase.

    Args:
        args (OmegaConf): Arguments
        train_queue (torch.utils.DataLoader): Training dataset.
        model (torch.nn.Module): The model that should be trained.
        criterion (callable): Loss that should be used for weight updates.
        optimizer: The optimizer that should be used for weight updates.

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
        data = Variable(data, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(data)
        loss = criterion(logits, target)

        if args.train.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.train.auxiliary_weight * loss_aux
        
        loss.backward()         # calculates dloss / dx for every parameter x
        nn.utils.clip_grad_norm_(model.parameters(), args.train.grad_clip)
        optimizer.step()        # performs gradient update for every x

        prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5))
        batch_size = data.size(0)
        objs.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        if step % args.run.report_freq == 0:
            logging.info(f"| Batch: {step:3d} | Loss: {objs.avg:5f} | Top1: {top1.avg:3f} | Top5: {top5.avg:3f} |")

    return top1.avg, objs.avg, top5.avg


if __name__ == '__main__':
    main()
