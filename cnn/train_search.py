import os
import sys
import time
import glob
import numpy as np
import torch
import train_utils
import aws_utils
import random
import copy
import logging
import hydra
import pickle
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import json
import io
import PIL

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import visualize

from timeit import default_timer as timer
from datetime import timedelta


@hydra.main(config_path="../configs/cnn/config.yaml", strict=False)
def main(args):
    """Performs NAS.
    
    Returns:
        str: Path to the json file where the genotype of the selected architecture was dumped to.

    """
    np.set_printoptions(precision=3)
    save_dir = os.getcwd()

    log = os.path.join(save_dir, "log.txt")

    # Setup SummaryWriter
    summary_dir = os.path.join(save_dir, "summary")
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    writer = SummaryWriter(summary_dir)

    # own writer that I use to keep track of interesting variables
    own_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    if args.run.s3_bucket is not None:
        aws_utils.download_from_s3(log, args.run.s3_bucket, log)

        train_utils.copy_code_to_experiment_dir("/home/julienf/git/gaea_release/cnn", save_dir)
        aws_utils.upload_directory(
            os.path.join(save_dir, "scripts"), args.run.s3_bucket
        )

    train_utils.set_up_logging(log)

    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    torch.cuda.set_device(args.run.gpu)
    logging.info("gpu device = %d" % args.run.gpu)
    logging.info("args = %s", args.pretty())

    # Set random seeds for random, numpy, torch and cuda
    rng_seed = train_utils.RNGSeed(args.run.seed)

    # Load respective architect
    if args.search.method in ["edarts", "gdarts", "eedarts"]:
        if args.search.fix_alphas:
            from architect.architect_edarts_edge_only import (
                ArchitectEDARTS as Architect,
            )
        else:
            from architect.architect_edarts import ArchitectEDARTS as Architect
    elif args.search.method in ["darts", "fdarts"]:
        from architect.architect_darts import ArchitectDARTS as Architect
    elif args.search.method == "egdas":
        from architect.architect_egdas import ArchitectEGDAS as Architect
    else:
        raise NotImplementedError

    # Load respective search spaces
    if args.search.search_space in ["darts", "darts_small"]:
        from search_spaces.darts.model_search import DARTSNetwork as Network
    elif "nas-bench-201" in args.search.search_space:
        from search_spaces.nasbench_201.model_search import (
            NASBENCH201Network as Network,
        )
    elif args.search.search_space == "pcdarts":
        from search_spaces.pc_darts.model_search import PCDARTSNetwork as Network
    else:
        raise NotImplementedError

    if args.train.smooth_cross_entropy:
        criterion = train_utils.cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    num_train, num_classes, train_queue, valid_queue = train_utils.create_data_queues(
        args
    )

    print("dataset: {}, num_classes: {}".format(args.run.dataset, num_classes))

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
    logging.info("param size = %fMB", train_utils.count_parameters_in_MB(model))
    own_writer.add_graph(model)

    optimizer, scheduler = train_utils.setup_optimizer(model, args)

    # TODO: separate args by model, architect, etc
    # TODO: look into using hydra for config files
    architect = Architect(model, args, writer)

    # Try to load a previous checkpoint
    try:
        start_epochs, history = train_utils.load(
            save_dir, rng_seed, model, optimizer, architect, args.run.s3_bucket
        )
        scheduler.last_epoch = start_epochs - 1
        (
            num_train,
            num_classes,
            train_queue,
            valid_queue,
        ) = train_utils.create_data_queues(args)
        logging.info('Resumed training from a previous checkpoint. Runtime measurement will be wrong.')
        train_start_time = 0
    except Exception as e:
        logging.info(e)
        start_epochs = 0
        train_start_time = timer()

    best_valid = 0
    epoch_best_valid = 0
    overall_visualization_time = 0 # don't count visualization into runtime
    for epoch in range(start_epochs, args.run.epochs):
        lr = scheduler.get_lr()[0]
        logging.info(f"\n| Epoch: {epoch:3d} / {args.run.epochs} | lr: {lr} |")

        model.drop_path_prob = args.train.drop_path_prob * epoch / args.run.epochs

        # training returns top1 and loss
        train_acc, train_obj, train_top5 = train(
            args, train_queue, valid_queue, model, architect, criterion, optimizer, lr,
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
        del genotype_graph_normal
        del binary_normal
        del stream_normal
        del graph_normal

        genotype_graph_reduce = visualize.plot(genotype.reduce, "", return_type="graph", output_format='png')
        binary_reduce = genotype_graph_reduce.pipe()
        stream_reduce = io.BytesIO(binary_reduce)
        graph_reduce = np.array(PIL.Image.open(stream_reduce).convert("RGB"))
        own_writer.add_image("Reduce_Cell", graph_reduce, epoch, dataformats="HWC")
        del genotype_graph_reduce
        del binary_reduce
        del stream_reduce
        del graph_reduce
        end_visualization = timer()
        overall_visualization_time += (end_visualization - start_visualization)

        if not args.search.single_level:
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

            if valid_acc > best_valid:
                best_valid = valid_acc
                best_genotype = architect.genotype()
                epoch_best_valid = epoch
            logging.info(f"| valid_acc: {valid_acc} |")

        train_utils.save(
            save_dir,
            epoch + 1,
            rng_seed,
            model,
            optimizer,
            architect,
            save_history=True,
            s3_bucket=args.run.s3_bucket,
        )

        scheduler.step()

    train_end_time = timer()
    logging.info(f"Visualization of cells during search took a total of {timedelta(seconds=overall_visualization_time)} (hh:mm:ss).")
    logging.info(f"This time is not included in the runtime given below.\n")
    logging.info(f"Training finished after {timedelta(seconds=((train_end_time - train_start_time) - overall_visualization_time))}(hh:mm:ss). Performing validation of final epoch...")
    valid_acc, valid_obj, valid_top5 = train_utils.infer(
        valid_queue,
        model,
        criterion,
        report_freq=args.run.report_freq,
        discrete=args.search.discrete,
    )

    own_writer.add_scalar('Loss/valid', valid_obj, args.run.epochs-1)
    own_writer.add_scalar('Top1/valid', valid_acc, args.run.epochs-1)
    own_writer.add_scalar('Top5/valid', valid_top5, args.run.epochs-1)

    if valid_acc > best_valid:
        best_valid = valid_acc
        best_genotype = architect.genotype()
        epoch_best_valid = args.run.epochs-1
    logging.info(f"| valid_acc: {valid_acc} |")

    logging.info(f"\nOverall best found genotype with validation accuracy of {best_valid} (found in epoch {epoch_best_valid}):")
    logging.info(f"{best_genotype}")

    # dump best genotype to json file, so that we can load it during evaluation phase (in train_final.py)
    genotype_dict = best_genotype._asdict()
    for key, val in genotype_dict.items():
        if type(val) == range:
            genotype_dict[key] = [node for node in val]
    if os.path.splitext(args.run.genotype_path)[1] != '.json':
        args.run.genotype_path += '.json'
    with open(args.run.genotype_path, 'w') as genotype_file:
        json.dump(genotype_dict, genotype_file, indent=4)

    logging.info(f"Search finished. Dumped best genotype into {args.run.genotype_path}")

    if args.run.s3_bucket is not None:
        filename = "cnn_genotypes.txt"
        aws_utils.download_from_s3(filename, args.run.s3_bucket, filename)

        with open(filename, "a+") as f:
            f.write("\n")
            f.write(
                "{}{}{}{} = {}".format(
                    args.search.search_space,
                    args.search.method,
                    args.run.dataset.replace("-", ""),
                    args.run.seed,
                    best_genotype,
                )
            )
        aws_utils.upload_to_s3(filename, args.run.s3_bucket, filename)
        aws_utils.upload_to_s3(log, args.run.s3_bucket, log)

    #return args.run.genotype_path


def train(
    args,
    train_queue,
    valid_queue,
    model,
    architect,
    criterion,
    optimizer,
    lr,
    random_arch=False,
):
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()

    for step, datapoint in enumerate(train_queue):
        # The search dataqueue for nas-bench-201  returns both train and valid data
        # when looping through queue.  This is disabled with single level is indicated.
        if "nas-bench-201" in args.search.search_space and not (
            args.search.single_level
        ):
            input, target, input_search, target_search = datapoint
        else:
            input, target = datapoint
            input_search, target_search = next(iter(valid_queue))

        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        # get a random minibatch from the search queue with replacement
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        model.train()

        # TODO: move architecture args into a separate dictionary within args
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
        # if random_arch or model.architect_type == "snas":
        #    architect.sample_arch_configure_model()

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
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.run.report_freq == 0:
            logging.info(f"| Train | Batch: {step:3d} | Loss: {objs.avg:e} | Top1: {top1.avg} | Top5: {top5.avg} |")

    return top1.avg, objs.avg, top5.avg


if __name__ == "__main__":
    main()
