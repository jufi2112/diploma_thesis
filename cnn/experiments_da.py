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
from search_spaces.pc_darts.model_search import PCDARTSNetwork as Network

@hydra.main(config_path="../configs/experiments_da/experiments_da.yaml", strict=False)
def main(args):
    """Performs either grid search over init_channels or employs a Gaussian process to search the best value for init_channels"""
    np.set_printoptions(precision=3)

    working_directory = os.getcwd()
    # TODO: dunno why original implementation used double logging
    log = os.path.join(working_directory, "overall_log.txt")
    train_utils.set_up_logging(log)

    logging.info("Hyperparameters:")
    logging.info(args.pretty())

    # prepare for search phase
    args.run = args.run_search_phase
    args.train = args.train_search_phase

    #logging.info("Search phase")
    #args.train = args.train_search
    #logging.info(args.pretty)
    #logging.info("Eval phase")
    #args.train = args.train_eval
    #logging.info(args.pretty)



def search_phase(args):
    """Performs NAS
    Code is mostly copied from train_search.py but modified to only work with GAEA PC-DARTS



    Returns:
        Genotype: Best found genotype.
        float: Runtime in seconds after which the best genotype was found.
        float: Train accuracy of the best found genotype.
        float: Validation accuracy of the best found genotype.
        float: Overall runtime of the search phase in seconds.
    """
    # Create folder structure
    base_dir = os.path.join(os.getcwd(), "search_phase_seed_" + str(args.run.seed))
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
    writer = SummaryWriter(summary_dir)
    # own writer that I use to keep track of interesting variables
    own_writer = SummaryWriter(tensorboard_dir)

    if not torch.cuda.is_available():
        logging.info("No GPU device available")
        sys.exit(-1)
    torch.cuda.set_device(args.run.gpu)

    # Set random seeds for random, numpy, torch and cuda
    rng_seed = train_utils.RNGSeed(args.run.seed)

    if args.train.smooth_cross_entropy:
        criterion = train_utils.cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

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
    model = Network(
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
        start_epochs, history, previous_runtime, best_accuracies = train_utils.load(
            checkpoint_dir,
            rng_seed,
            model,
            optimizer,
            architect,
            args.run.s3_bucket
        )
        scheduler.last_epoch = start_epochs - 1
        if best_accuracies is None:
            best_accuracies = {
                "train": 0.0,           # for single-level search, used to keep track of best genotype
                "valid": 0.0,           # for bi-level search, used to keep track of best genotype
                "epoch": 0,             # epoch the best accuracy was observed
                "genotype_raw": None,   # best genotype
                "genotype_dict": None,  # best genotype stored as dict (for serialization)
                "runtime": 0.0          # runtime after which the best genotype was found
            }
        else:
            best_accuracies['genotype_raw'] = _dict_to_genotype(best_accuracies['genotype_dict'])
        logging.info("Resumed training from a previous checkpoint.")
    except Exception as e:
        logging.info(e)
        start_epochs = 0
        previous_runtime = 0

        best_accuracies = {
            "train": 0.0,           # for single-level search, used to keep track of best genotype
            "valid": 0.0,           # for bi-level search, used to keep track of best genotype
            "epoch": 0,             # epoch the best accuracy was observed
            "genotype_raw": None,   # best genotype
            "genotype_dict": None,  # best genotype stored as dict (for serialization)
            "runtime": 0.0          # runtime after which the best genotype was found
        }
    
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

        if (args.search.single_level and train_acc > best_accuracies['train']) or (not args.search.single_level and valid_acc > best_accuracies['valid']):
                best_accuracies['train'] = train_acc
                best_accuracies['valid'] = valid_acc
                best_accuracies['epoch'] = epoch
                best_accuracies['genotype_raw'] = genotype
                best_accuracies['genotype_dict'] = _genotype_to_dict(genotype)
                best_accuracies['runtime'] = timer() - train_start_time - overall_visualization_time + previous_runtime

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
            best_accuracies=best_accuracies
        )

        scheduler.step()

    train_end_time = timer()
    overall_runtime = train_end_time - train_start_time - overall_visualization_time + previous_runtime
    logging.info(f"Visualization of cells during search took a total of {timedelta(seconds=overall_visualization_time)} (hh:mm:ss).")
    logging.info(f"This time is not included in the runtime given below.\n")
    logging.info(f"Training finished after {timedelta(seconds=overall_runtime)}(hh:mm:ss).")

    if args.search.single_level:
        logging.info((
            f"\nBest genotype according to training accuracy found in epoch {best_accuracies["epoch"]} after "
            f"{timedelta(seconds=best_accuracies['runtime'])} hh:mm:ss"
        ))
    else:
        logging.info((
            f"\nBest genotype according to validation accuracy found in epoch {best_accuracies["epoch"]} after "
            f"{timedelta(seconds=best_accuracies['runtime'])} hh:mm:ss"
        ))
    logging.info(f"Train accuracy: {best_accuracies['train']}")
    logging.info(f"Valid accuracy: {best_accuracies['valid']}")
    logging.info(f"Genotype: {best_accuracies['genotype_raw']}")

    # dump best genotype to json file, so that we can load it during evaluation phase
    genotype_file_path = os.path.join(genotype_dir, "genotype_init_channels_" + str(args.train.init_channels) + ".json")
    with open(genotype_file_path, 'w') as genotype_file:
        json.dump(best_accuracies['genotype_dict'], genotype_file, indent=4)

    logging.info(f"Search finished. Dumped best genotype into {genotype_file_path}")

    # before return, remove logging filehandler, so that the following logs aren't written in the current log
    logging.getLogger().removeHandler(logging.getLogger().handlers[-1])
    return best_accuracies['genotype_raw'], best_accuracies['runtime'], best_accuracies['train'], best_accuracies['valid'], overall_runtime


def _genotype_to_dict(genotype: namedtuple):
    """Converts the given genotype to a dictionary that can be serialized.
    Inverse operation to _dict_to_genotype().

    Args:
        genotype (namedtuple): The genotype that should be converted.

    Returns:
        dict: The converted genotype.
    """
    genotype_dict = genotype._asdict()
    for key, val in genotype_dict.items():
        if type(val) == range:
            genotype_dict[key] = [node for node in val]
    return genotype_dict


def _dict_to_genotype(genotype_dict: dict):
    """Converts the given dict to a genotype.
    Inverse operation to _genotype_to_dict().

    Args:
        genotype_dict (dict): A genotype represented as dict.

    Returns:
        namedtuple: The dict converted to a genotype.
    """
    genotype = Genotype(
        normal=genotype_dict['normal'],
        normal_concat=genotype_dict['normal_concat'],
        reduce=genotype_dict['reduce'],
        reduce_concat=genotype_dict['reduce_concat']
    )
    return genotype


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
        args: Arguments
        train_queue (torch.utils.DataLoader): Training dataset.
        valid_queue (torch.utils.DataLoader): Validation dataset.
            When utilizing single-level search, this is supposed to be a 
                DataLoader that also points to the training data and !!NOT!! the actual validation data.
            When utilizing bi-level search, this is supposed to be a 
                DataLoader that points to the validation data.
        model (nn.Module): The model that should be trained.
        architect (Architect): Architect that should be used to update architecture parameters.
        criterion (callable): Loss that should be utilized for weight updates.
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


if __name__ == '__main__':
    main()
