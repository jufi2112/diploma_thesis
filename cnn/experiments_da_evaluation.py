"""Used to evaluate a given trained architecture on the CIFAR-10 test dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import train_utils
from timeit import default_timer as timer
from datetime import timedelta
import hydra
import logging
import json
import sys
import os

from collections import namedtuple
Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

from search_spaces.darts.model import NetworkCIFAR

@hydra.main(config_path="../configs/experiments_da/testdata.yaml", strict=False)
def main(args):

    logging.info(f"Parameters: \n{args.pretty()}")

    torch.cuda.set_device(args.run.gpu)
    torch.cuda.empty_cache()
    current_device = torch.cuda.current_device()
    logging.info(f"Current cuda device: {current_device} - {torch.cuda.get_device_name(current_device)}")
    torch.cuda.reset_peak_memory_stats()

    rng_seed = train_utils.RNGSeed(args.run.seed)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.run.gpu)

    _, _, test_dataset = train_utils.get_cifar10_data_sets(args)

    test_queue = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.train.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0
    )

    genotype_dir = os.path.join(args.run.checkpoint_path, 'genotypes')
    checkpoint_dir = os.path.join(args.run.checkpoint_path, 'checkpoints')

    utilized_seeds = [int(os.listdir(genotype_dir)[i].split('_')[2]) for i in len(os.listdir(genotype_dir))]

    genotypes = {
        seed: os.path.join(genotype_dir, f'genotype_seed_{seed}.json')
        for seed in utilized_seeds
    }
    
    checkpoints_to_test = os.path.listdir(checkpoint_dir)

    for checkpoint in checkpoints_to_test:
        torch.cuda.empty_cache()
        logging.info(f"Checkpoint being tested: {checkpoint}")
        seed = int(checkpoint.split('_')[6])
        zeta_eval = int(checkpoint.split('_')[4])

        # Load genotype
        with open(genotypes[seed], 'r') as genotype_file:
            genotype_dict = json.load(genotype_file)
        genotype = train_utils.dict_to_genotype(genotype_dict)

        model = NetworkCIFAR(
            zeta_eval,
            10,
            args.architecture.layers,
            args.architecture.auxiliary,
            genotype
        )

        model = model.cuda(args.run.gpu)

        # Load weights from checkpoint
        if args.run.gpu is not None:
            map_location = {'cuda:0': f'cuda:{args.run.gpu}'}
        ckpt = torch.load(os.path.join(checkpoint_dir, checkpoint)) if args.run.gpu is None else torch.load(os.path.join(checkpoint_dir, checkpoint), map_location=map_location)
        model.load_states(ckpt['model'])

        infer_timer = timer()

        test_acc, test_obj, test_top5 = train_utils.infer(
            test_queue,
            model,
            criterion,
            args.run.report_freq
        )

        infer_finished = timer() - infer_timer

        #logging.info(f"    Genotype: {args.architecture.genotype}")
        logging.info(f"    Test accuracy: {test_acc}")
        logging.info(f"    Test loss: {test_obj}")
        logging.info(f"    Test top5 accuracy: {test_top5}")
        logging.info(f"    Time needed for inference: {timedelta(seconds=infer_finished)}")

        del model

    sys.exit(0)


if __name__ == "__main__":
    main()