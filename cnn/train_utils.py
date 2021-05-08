import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import sys
from distutils.dir_util import copy_tree
import aws_utils
import pickle
from lr_schedulers import *
from copy import deepcopy
from datetime import timedelta

# from genotypes import PRIMITIVES
import logging
import torchvision.datasets as dset
from torch.autograd import Variable
from collections import deque
import torchvision.transforms as transforms

# Import AutoDL Nasbench-201 functions for data loading
from lib.datasets.get_dataset_with_transform import get_datasets, get_nas_search_loaders

from collections import namedtuple
from genotypes_to_visualize import Genotype


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.train.cutout:
        train_transform.transforms.append(Cutout(args.train.cutout_length))
    print(train_transform.transforms)

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )

    return train_transform, valid_transform, test_transform


def count_parameters_in_MB(model):
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


def drop_path(x, drop_prob):
    """For every element along the first dimension of x, returns either the
    original element or 0, according to drop_prob.

    Args:
        x (torch.cuda.Tensor): Torch cuda tensor where elements along the first
            dimension should be zeroed.
        drop_prob (float): The probability with which elements along the first
            dimension of x should be zeroed. A value of 0 means that x is
            returned unmodified, whereas a value of 1 zeroes all elements.

    Returns:
        torch.cuda.Tensor: A tensor of the same shape as the input, with elements
            of either the original ones or 0.
    """
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = Variable(
            torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        )
        # mask contains for every element inside the batch a float of either
        # 0 or 1 
        x.div_(keep_prob) # so that mul will result in the original value
        x.mul_(mask)
    return x


def set_up_logging(path):
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def label_smoothing(pred, target, eta=0.1):
    """
    Code from https://github.com/mit-han-lab/ProxylessNAS/blob/master/training/utils/bags_of_tricks.py
    See https://arxiv.org/pdf/1512.00567.pdf for more info about label smoothing softmax.

    Args:
      pred: predictions
      target: ints representing class label
      eta: smoothing parameter, eta is spread to other classes
    Return:
      same shape as predictions
    """
    n_classes = pred.size(1)
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_classes * 1


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def cross_entropy_with_label_smoothing(pred, target, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta)
    return cross_entropy_for_onehot(pred, onehot_target)


def setup_optimizer(model, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.train.learning_rate,
        momentum=args.train.momentum,
        weight_decay=args.train.weight_decay,
    )

    min_lr = args.train.learning_rate_min
    if args.train.scheduler == "triangle":
        scheduler = TriangleScheduler(
            optimizer,
            0.003,
            0.1,
            min_lr,
            5,
            args.run.epochs,
            backoff_scheduler=scheduler,
        )
    else:
        lr_anneal_cycles = 1
        if "lr_anneal_cycles" in args.train:
            lr_anneal_cycles = args.train.lr_anneal_cycles
        if args.train.scheduler == "cosine":
            scheduler = CosinePowerAnnealing(
                optimizer, 1, lr_anneal_cycles, min_lr, args.run.scheduler_epochs
            )
        elif args.train.scheduler == "powercosine":
            scheduler = CosinePowerAnnealing(
                optimizer, 2, lr_anneal_cycles, min_lr, args.run.scheduler_epochs
            )
        else:
            raise NotImplementedError(
                "lr scheduler not implemented, please select one from [cosine, powercosine, triangle]"
            )

    return optimizer, scheduler


def create_nasbench_201_data_queues(args, eval_split=False):
    assert args.run.dataset in ["cifar10", "cifar100", "ImageNet16-120"]
    path_mapping = {
        "cifar10": "cifar-10-batches-py",
        "cifar100": "cifar-100-python",
        "ImageNet16-120": "ImageNet16",
    }
    train_data, valid_data, xshape, class_num = get_datasets(
        args.run.dataset,
        os.path.join(args.run.data, path_mapping[args.run.dataset]),
        -1,
    )
    if eval_split:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train.batch_size,
            shuffle=True,
            num_workers=args.run.n_threads_data,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train.batch_size,
            shuffle=False,
            num_workers=args.run.n_threads_data,
            pin_memory=True,
        )
        return len(train_data), class_num, train_loader, valid_loader

    if args.search.single_level:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train.batch_size,
            shuffle=True,
            num_workers=args.run.n_threads_data,
            pin_memory=True,
        )
        valid_data = deepcopy(train_data)
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.run.n_threads_data,
        )
        num_train = len(train_loader.dataset)
        return num_train, class_num, train_loader, valid_loader

    search_loader, _, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        args.run.dataset,
        os.path.join(args.run.autodl, "configs/nas-benchmark/"),
        args.train.batch_size,
        args.run.n_threads_data,
    )
    num_train = len(search_loader.dataset)
    return num_train, class_num, search_loader, valid_loader


def create_cifar10_data_queues_own(args, evaluation_mode=False):
    """Creates and returns CIFAR-10 train, validation and test data sets for experiments.
    This is a modification of the create_data_queues method more specifically tailored to the needs of my diploma thesis.
    During single level search, two train datasets with the same images are returned to enable usage of the existing code

    Args:
        args: Arguments
        evaluation_mode (bool): Whether the data sets are used for architecture search or architecture evaluation.
            During evaluation, only a small portion of the training data should be utilized for validation
                (based on train.train_portion in eval.yaml)
            During search, two possibilities exist:
                single_level: Only a small portion of the training data should be utilized for validation
                    (based on search.train_portion_single_level in method_eedarts_space_pcdarts.yaml)
                bi-level: Train and validation sets are splitted according to 
                    search.train_portion_bi_level in method_eedarts_space_pcdarts.yaml
        
    Returns:
        int: Number of classes.
        (DataLoader, DataLoader): Tuple that contains two data loaders for the same training data.
            Second data loader is None during bi-level search or when evaluation mode is true.
        DataLoader: The validation dataset.
        DataLoader: The test dataset.
        (int, int, int): Tuple that contains the number of training, validation and test images.
    """
    if "nas-bench-201" in args.search.search_space:
        raise ValueError("This function is not designed for NAS-Bench-201, use create_data_queues() instead. (Care different meaning of validation set though)")
    train_transform, valid_transform, test_transform = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.run.data, train=True, download=True, transform=train_transform
    )
    valid_data = dset.CIFAR10(
        root=args.run.data, train=True, download=True, transform=valid_transform
    )
    test_data = dset.CIFAR10(
        root=args.run.data, train=False, download=True, transform=test_transform
    )
    num_train = len(train_data) # used to calculate splits for train and validation sets
    train_indices = list(range(num_train))
    number_train_images = 0 # used to simply keep track of how many training images there are
    number_valid_images = 0
    number_test_images = len(test_data)
    if evaluation_mode:
        # Evaluating a single architecture
        #np.random.shuffle(train_indices)
        train_end = int(np.floor(num_train * args.train.train_portion))
        number_train_images = train_end
        number_valid_images = num_train - number_train_images

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices[:train_end])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices[train_end:])

        train_queue = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=args.train.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=args.run.n_threads_data
        )
        train_2_queue = None
        valid_queue = torch.utils.data.DataLoader(
            dataset=valid_data,
            batch_size=args.train.batch_size,
            sampler=valid_sampler,
            pin_memory=True,
            num_workers=args.run.n_threads_data
        )
        test_queue = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=args.train.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.run.n_threads_data
        )
    else:
        # Architecture search
        if args.search.single_level:
            split = int(np.floor(num_train * args.search.train_portion_single_level))
            train_end = split
            train_2_end = split
            valid_start = split
        else:
            valid_data = deepcopy(train_data) # want to have the exact same data (including preprocessing) as train_data
            split = int(np.floor(num_train * args.search.train_portion_bi_level))
            train_end = split
            train_2_end = None  # don't need second train data loader for bi-level, since we use validation set for this
            valid_start = split

        number_train_images = train_end
        number_valid_images = num_train - valid_start

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices[:train_end])
        train_2_sampler = None if train_2_end == None else torch.utils.data.sampler.SubsetRandomSampler(train_indices[:train_2_end]) # only used during single level search
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices[valid_start:])

        train_queue = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=args.train.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=0
        )
        train_2_queue = None if train_2_end == None else torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=args.train.batch_size,
            sampler=train_2_sampler,
            pin_memory=True,
            num_workers=0
        )
        valid_queue = torch.utils.data.DataLoader(
            dataset=valid_data,
            batch_size=args.train.batch_size,
            sampler=valid_sampler,
            pin_memory=True,
            num_workers=0
        )
        test_queue = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=args.train.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )
    return 10, (train_queue, train_2_queue), valid_queue, test_queue, (number_train_images, number_valid_images, number_test_images)


def create_data_queues(args, eval_split=False):
    """Creates training and validation data queues.
    When eval_split is set to true, returns test data instead of validation data.

    Args:
        args: Arguments
        eval_split (bool): True if evaluating a single architecture from scratch (evaluation phase), false during search.
            If set to true, returns test data instead of validation data
    """
    if "nas-bench-201" in args.search.search_space:
        return create_nasbench_201_data_queues(args, eval_split)

    train_transform, valid_transform, _ = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.run.data, train=True, download=True, transform=train_transform
    )

    # These are TEST IMAGES !!! not validation images
    valid_data = dset.CIFAR10(
        root=args.run.data, train=False, download=True, transform=valid_transform
    )
    num_train = 64 * 10 if args.run.test_code else len(train_data)

    if eval_split:
        # This is used if evaluating a single architecture from scratch.
        train_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.run.n_threads_data,
        )

        # will contain test images
        valid_queue = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.run.n_threads_data,
        )
    else:
        # This is used during architecture search.
        indices = list(range(num_train))

        if args.search.single_level:
            train_end = num_train
            val_start = 0
        else:
            split = int(np.floor(args.search.train_portion * num_train))
            train_end = split
            val_start = split

        valid_data = deepcopy(train_data)

        train_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[0:train_end]),
            pin_memory=True,
            num_workers=0,
        )

        valid_queue = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                np.random.permutation(indices[val_start:])
            ),
            pin_memory=True,
            num_workers=0,
        )
    return num_train, 10, train_queue, valid_queue


class RNGSeed:
    def __init__(self, seed):
        self.seed = seed
        self.set_random_seeds()

    def set_random_seeds(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = False
        torch.manual_seed(seed)
        cudnn.enabled = True
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    def get_save_states(self):
        rng_states = {
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all(),
        }
        return rng_states

    def load_states(self, rng_states):
        random.setstate(rng_states["random_state"])
        np.random.set_state(rng_states["np_random_state"])
        torch.set_rng_state(rng_states["torch_random_state"])
        torch.cuda.set_rng_state_all(rng_states["torch_cuda_random_state"])


def save(
    folder,
    epochs,
    rng_seed,
    model,
    optimizer,
    architect=None,
    save_history=False,
    s3_bucket=None,
    runtime=0.0,
    best_observed=None,
    best_eval=False
):
    """
    Create checkpoint and save to directory.
    Overwriting of previous checkpoints is done atomic by first creating a
        'buffer' file and then using os.replace. See also https://github.com/PyTorchLightning/pytorch-lightning/pull/689
    TODO: should remove s3 handling and have separate class for that.

    Args:
        folder: save directory
        epochs: number of epochs completed
        rng_state: rng states for np, torch, random, etc.
        model: model object with its own get_save_states method
        optimizer: model optimizer
        architect: architect object with its own get_save_states method
        s3_bucket: s3 bucket name for saving to AWS s3
        runtime (float): Current runtime of the method as given by timeit.timer
        best_observed (dict): Dictionary that keeps track of the currently best observed genotype and its properties.
            This is also used during evaluation (with genotype related values set to None).
        best_eval (bool): Whether the checkpoint is created for the best currently observed weights during evaluation.
            This means that the checkpoint is saved with a dedicated name (model_best.ckpt).
    """

    checkpoint = {
        "epochs": epochs,
        "rng_seed": rng_seed.get_save_states(),
        "optimizer": optimizer.state_dict(),
        "model": model.get_save_states(),
        "runtime": runtime
    }

    if architect is not None:
        checkpoint["architect"] = architect.get_save_states()

    if best_observed is not None:
        checkpoint['best_observed'] = {
            'train': best_observed['train'],
            'valid': best_observed['valid'],
            'epoch': best_observed['epoch'],
            'genotype_dict': best_observed['genotype_dict'],
            'runtime': best_observed['runtime']
        }

    # if you change path of model_best.ckpt, also change the return value of experiments_da.py.evaluation_phase()
    ckpt = os.path.join(folder, "model_best.ckpt") if best_eval else os.path.join(folder, "model.ckpt")
    # needed for atomic overwriting
    ckpt_part = ckpt + ".part"
    torch.save(checkpoint, ckpt_part)
    os.replace(ckpt_part, ckpt)


    history = None
    if save_history:
        history_file = os.path.join(folder, "history.pkl")
        history_file_part = history_file + ".part"
        history = architect.get_history()
        with open(history_file_part, "wb") as f:
            pickle.dump(history, f)
        # atomic replace
        os.replace(history_file_part, history_file)

    if s3_bucket is not None:
        log = os.path.join(folder, "log.txt")
        aws_utils.upload_to_s3(ckpt, s3_bucket, ckpt)
        aws_utils.upload_to_s3(log, s3_bucket, log)
        if history is not None:
            aws_utils.upload_to_s3(history_file, s3_bucket, history_file)
        # try:
        #    summary_dir = os.path.join(folder, 'summary')
        #    for root, dirs, files in os.walk(summary_dir):
        #        for f in files:
        #            path = os.path.join(root, f)
        #            aws_utils.upload_to_s3(path, s3_bucket, path)


def load(
    folder, 
    rng_seed, 
    model, 
    optimizer, 
    architect=None, 
    s3_bucket=None,
    best_eval=False
):
    """Loads checkpoint

    Args:
        folder (str): Directory that contains the checkpoint which should be loaded.
        rng_seed (RNGSeed): Random seed object that should get initialized from the checkpoint.
            Reference is modified.
        model: Model that should get initialized from the checkpoint.
            Reference is modified.
        optimizer: Optimizer that should get initialized from the checkpoint.
            Reference is modified.
        architect: Architectut that should get initialized from the checkpoint.
            Reference is modified.
        s3_bucket: AWS stuff. Unused.
        best_eval (bool): Whether the best checkpoint from evaluation should be loaded.
            This will search for a checkpoint called 'model_best.ckpt' instead of 'model.ckpt'.
            If no such file is found, an error is raised.

    Returns:
        int: Epochs
        history
        int: Current overall runtime of the model
        dict: Properties of the currently best observed genotype
    """
    # Try to download log and ckpt from s3 first to see if a ckpt exists.
    ckpt = os.path.join(folder, "model_best.ckpt") if best_eval else os.path.join(folder, "model.ckpt")

    if not os.path.isfile(ckpt):
        raise ValueError(f"No valid checkpoint file found: {ckpt}")

    history_file = os.path.join(folder, "history.pkl")
    history = {}

    if s3_bucket is not None:
        aws_utils.download_from_s3(ckpt, s3_bucket, ckpt)
        try:
            aws_utils.download_from_s3(history_file, s3_bucket, history_file)
        except:
            logging.info("history.pkl not in s3 bucket")

    if os.path.exists(history_file) and architect is not None:
        with open(history_file, "rb") as f:
            history = pickle.load(f)
        # TODO: update architecture history
        architect.load_history(history)

    checkpoint = torch.load(ckpt)

    epochs = checkpoint["epochs"]
    rng_seed.load_states(checkpoint["rng_seed"])
    model.load_states(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "runtime" in checkpoint.keys():
        runtime = checkpoint["runtime"]
    else:
        runtime = 0
    if "best_observed" in checkpoint.keys():
        best_observed = checkpoint["best_observed"]
        best_observed["genotype_raw"] = dict_to_genotype(best_observed["genotype_dict"])
    else:
        best_observed = None
    if architect is not None:
        architect.load_states(checkpoint["architect"])

    #logging.info(f"Resumed model trained for {epochs} epochs")
    #logging.info(f"Resumed model trained for {timedelta(seconds=runtime)} hh:mm:ss")

    return epochs, history, runtime, best_observed


def infer(valid_queue, model, criterion, report_freq=50, discrete=False):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    # if arch is not None:
    #    alphas_normal, alphas_reduce = get_weights_from_arch(model, arch)
    #    normal_orig = torch.tensor(model.alphas_normal.data, requires_grad=False)
    #    reduce_orig = torch.tensor(model.alphas_reduce.data, requires_grad=False)
    #    model.alphas_normal.data = alphas_normal
    #    model.alphas_reduce.data = alphas_reduce
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            # if model.__class__.__name__ == "NetworkCIFAR":
            #    logits, _ = model(input)
            # else:
            #     logits, _ = model(input, discrete=model.architect_type in ["snas", "fixed"])
            #     TODO: change to also eval best discrete architecture
            logits, _ = model(input, **{"discrete": discrete})
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % report_freq == 0:
                logging.info(f"| Validation | Batch {step:3d} | Loss: {objs.avg:e} | Top1: {top1.avg} | Top5: {top5.avg} |")
    # if arch is not None:
    #    model.alphas_normal.data.copy_(normal_orig)
    #    model.alphas_reduce.data.copy_(reduce_orig)

    return top1.avg, objs.avg, top5.avg


def copy_code_to_experiment_dir(code, experiment_dir):
    scripts_dir = os.path.join(experiment_dir, "scripts")

    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)
    copy_tree(code, scripts_dir)


def create_exp_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))


def genotype_to_dict(genotype: namedtuple):
    """Converts the given genotype to a dictionary that can be serialized.
    Inverse operation to dict_to_genotype().

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


def dict_to_genotype(genotype_dict: dict):
    """Converts the given dict to a genotype.
    Inverse operation to genotype_to_dict().

    Args:
        genotype_dict (dict): A genotype represented as dict.

    Returns:
        namedtuple: The dict converted to a genotype.
    """
    if genotype_dict is None:
        return None
    genotype = Genotype(
        normal=genotype_dict['normal'],
        normal_concat=genotype_dict['normal_concat'],
        reduce=genotype_dict['reduce'],
        reduce_concat=genotype_dict['reduce_concat']
    )
    return genotype


def save_outer_loop_checkpoint(folder, history: dict, overall_runtime: float):
    """Saves a checkpoint of the given outer loop history into the provided folder.
    Saving is done in an atomic way by first creating a .part file which is then
    renamed with os.replace().

    Args:
        folder (str): Directory in which to save the checkpoint.
        history: History that should be saved. Needs to be serializable by
            torch.save() function
        overall_runtime (float): Runtime of the given history in seconds.
    """
    checkpoint = {
        'history': history,
        'runtime': runtime
    }
    ckpt = os.path.join(folder, 'outer_loop.ckpt')
    ckpt_part = ckpt + ".part"
    torch.save(checkpoint, ckpt_part)
    os.replace(ckpt_part, ckpt)


def load_outer_loop_checkpoint(folder):
    """Loads the outer loop history checkpoint from the provided folder.

    Args:
        folder (str): Path to the folder that contains the search history checkpoint.

    Returns:
        dict: The outer loop history.
        float: Overall runtime of the loaded outer loop in seconds.
    """
    ckpt = os.path.join(folder, 'outer_loop.ckpt')
    checkpoint = torch.load(ckpt)

    return checkpoint['history'], checkpoint['runtime']

    

    