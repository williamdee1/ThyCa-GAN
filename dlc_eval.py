import json
import argparse
import torch.nn as nn
import torch
import logging
from modules.data_funcs import dataloader
from modules.utils import model_init, generate_id, gpu_init
from modules.test_eval import test_eval


def eval_dlc():
    """
    Evaluates a pre-trained model against a given test set containing thyroid histopathology images.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', help='Directory containing input image dataset', required=True)
    parser.add_argument('--labels', help='File name for labels json file', required=True)
    parser.add_argument('--out_dir', help='Output directory for results', required=True)
    parser.add_argument('--split_file', help='Data split json file', required=True)
    parser.add_argument('--run_id', help='Unique id for training run', required=True)
    parser.add_argument('--mdl_loc', help='.pth file containing pre-trained model weights', required=True)
    parser.add_argument('--batch_size', help='Data batch size within epochs', default=20, type=int)
    parser.add_argument('--workers', help='No. workers', default=8, type=int)
    parser.add_argument('--crop_size', help='Size of cropped images', default=512, type=int)
    parser.add_argument('--model_type', help='Pre-trained model to load', default='resnet101', type=str)
    opt = parser.parse_args()

    # ----------------
    #  Initialization
    # ----------------
    model = model_init(opt.model_type)                                              # Pre-trained model
    criterion = nn.CrossEntropyLoss()                                               # Loss function
    data_labels = json.load(open(opt.labels))                                       # Load data labels
    run_dir = generate_id(opt.out_dir, opt.run_id)                                  # Create a unique dir for output
    log_path = "%s/eval_log.txt" % run_dir                                          # Save logs to run_dir
    logging.basicConfig(filename=log_path, level=logging.INFO)                      # Initialize logging
    logging.info(opt)                                                               # Record params in log file

    # Assign model to GPU if present:
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        model, criterion, device = gpu_init(model, criterion)
    else:
        device = "cpu"

    # ------------------
    #  Model Evaluation
    # ------------------
    logging.info("[------ Beginning Model Evaluation -------]")

    # Initialise test data loader:
    test_loader, _ = dataloader(opt.src_dir, data_labels['meta'], opt.split_file, 'test',
                                opt.crop_size, opt.batch_size, opt.workers, None, opt.run_id)

    # Perform model evaluation on test set:
    test_eval(model, opt.mdl_loc, test_loader, device, run_dir)


if __name__ == "__main__":
    eval_dlc()
