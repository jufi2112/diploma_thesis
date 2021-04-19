"""
Evaluation routine for training final architectures from DARTS search space.
Do not use for NASBENCH-201 or NASBENCH-1SHOT1 search spaces.
Adapted version of train_aws.py
"""

import os
import sys
import time
import numpy as np
import random
import torch
import train_utils
import logging
import argparse
import json
from timeit import default_timer as timer
from datetime import timedelta

import hydra
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from search_spaces.darts.model import NetworkCIFAR as Network
from search_spaces.darts.genotypes import Genotype

@hydra.main(config_path="../configs/cnn/config_final_training.yaml", strict=False)
def main(args):
    """Evaluates an architecture by completely training it"""

    log = os.path.join(os.getcwd(), 'log_architecture_evaluation.txt')
    train_utils.set_up_logging(log)

    CIFAR_CLASSES = 10

    if not torch.cuda.is_available():
        logging.error("No GPU device available!")
        sys.exit(1)

    rng_seed = train_utils.RNGSeed(args.run.seed)
    torch.cuda.set_device(args.run.gpu)
    logging.info(f"Training hyperparameters: TODO: print arguments for training")
    logging.info(f"    GPU device: {args.run.gpu}")
    logging.info(f"    Number of cells to stack: {args.train.layers}")
    logging.info(f"    Initial number of channels per cell: {args.train.init_channels}")
    logging.info(f"    Training epochs: {args.run.epochs}")
    logging.info(f"    Batch size: {args.train.batch_size}")
    logging.info(f"    Loss function: Cross entropy loss")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # load genotype that should be evaluated
    with open(args.run.genotype_path) as genotype_file:
        genotype_dict = json.load(genotype_file)

    # create namedtuple from dict
    genotype = Genotype(
        normal=genotype_dict['normal'],
        normal_concat=genotype_dict['normal_concat'],
        reduce=genotype_dict['reduce'],
        reduce_concat=genotype_dict['reduce_concat']
    )

    logging.info(f"Evaluation phase started for genotype: \n{genotype}")

    model = Network(
        args.train.init_channels,
        CIFAR_CLASSES,
        args.train.layers,
        args.train.auxiliary,
        genotype
    )
    model = model.cuda()

    optimizer, scheduler = train_utils.setup_optimizer(model, args)

    logging.info(f"Size of model parameters: {train_utils.count_parameters_in_MB(model)} MB")
    total_params = sum(x.data.nelement() for x in model.parameters())
    logging.info(f"Total parameters of model: {total_params}")

    # check if we already trained
    try:
        start_epochs, _ = train_utils.load(
            os.getcwd(), rng_seed, model, optimizer, s3_bucket=None
        )
        scheduler.last_epoch = start_epochs - 1
    except Exception as e:
        print(e)
        start_epochs = 0

    num_train, num_classes, train_queue, valid_queue = train_utils.create_data_queues(
        args, eval_split=True
    )

    train_start_time = timer()

    for epoch in range(start_epochs, args.run.epochs):
        logging.info(f"| Epoch: {epoch+1:4d}/{args.run.epochs} | lr: {scheduler.get_lr()[0]} |")
        model.drop_path_prob = args.train.drop_path_prob * epoch / args.run.epochs

        train_acc, train_obj = train(args, train_queue, model, criterion, optimizer)
        logging.info(f"| train_acc: {train_acc} |")

        valid_acc, valid_obj = train_utils.infer(
            valid_queue, model, criterion, report_freq=args.run.report_freq
        )
        logging.info(f"| valid_acc: {valid_acc} |")

        train_utils.save(
            os.getcwd(), epoch+1, rng_seed, model, optimizer, s3_bucket=None
        )
        scheduler.step()

    train_end_time = timer()
    logging.info(f"\nTraining finished after {timedelta(seconds=(train_end_time - train_start_time))} (hh:mm:ss).")


def train(args, train_queue, model, criterion, optimizer):
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
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train.grad_clip)
        optimizer.step()

        prec1, prec5 = train_utils.accuracy(logits, target, topk=(1,5))
        batch_size = data.size(0)
        objs.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        if step % args.run.report_freq == 0:
            logging.info(f"| Batch: {step:3d} | Loss: {objs.avg} | Top1: {top1.avg} | Top5: {top5.avg} |")
        
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()