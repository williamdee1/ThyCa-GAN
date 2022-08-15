import torchvision.models as models
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import re
import torch
import logging


def model_init(model_type):
    """ Loads a pre-trained model from torchvision """
    # Importing pre-trained vision model:
    model = getattr(models, model_type)(pretrained=True)
    # Altering final layer of the model to output a binary prediction - PTC or Non-PTC
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model


def gpu_init(model, criterion):
    gpu_count = torch.cuda.device_count()
    logging.info("Using %s GPU(s)..." % gpu_count)
    if gpu_count > 1:
        dev_ids = list(range(0, torch.cuda.device_count()))
        para_model = nn.DataParallel(model, device_ids=dev_ids)                     # Create parallelized model
        device = 0
        para_model.to(device)                                                       # Assigning model output to device:0
        criterion.to(device)

        return para_model, criterion, device

    else:
        model.cuda()
        criterion.cuda()
        device = "cuda:0"

        return model, criterion, device


def sched_init(optimizer, lrd_epc, lrd_fac):
    """ Initializes the learning rate scheduler """
    lr_scheduler = ReduceLROnPlateau(
                                    optimizer=optimizer,
                                    mode='min',
                                    factor=lrd_fac,
                                    patience=lrd_epc,
                                    verbose=True,
                                )

    return lr_scheduler


def generate_id(out_dir, run_id):
    """
    Generates a unique directory for the query results.
    Source:  https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py
    """
    prev_run_dirs = []
    if os.path.isdir(out_dir):
        prev_run_dirs = [x for x in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(out_dir, f'{cur_run_id:03d}-{run_id}')

    # Creates the directory if it doesn't already exist:
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    return run_dir
