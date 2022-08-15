import json
import argparse
import torch.nn as nn
import torch
import logging
from shutil import rmtree
from modules.data_funcs import dataloader
from modules.utils import model_init, sched_init, generate_id, gpu_init
from modules.training import train_loop


def train_dlc():
    """
    Trains a deep learning model to classify thyroid histopathology images as papillary thyroid carcinoma-like
    (PTC-like) or not. Evaluates the model against a given test set.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', help='Directory containing input image dataset', required=True)
    parser.add_argument('--labels', help='File name for labels json file', required=True)
    parser.add_argument('--out_dir', help='Output directory for results', required=True)
    parser.add_argument('--split_file', help='Data split json file', required=True)
    parser.add_argument('--run_id', help='Unique id for training run', required=True)
    parser.add_argument('--gan_params', help='Parameters for GAN-Generation', default=None)
    parser.add_argument('--lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--epochs', help='No. epochs to train model', default=500, type=int)
    parser.add_argument('--batch_size', help='Data batch size within epochs', default=8, type=int)
    parser.add_argument('--workers', help='No. workers', default=8, type=int)
    parser.add_argument('--crop_size', help='Size of cropped images', default=512, type=int)
    parser.add_argument('--model_type', help='Pre-trained model to load', default='resnet101', type=str)
    parser.add_argument('--lrd_epc', help='Reduce learning rate after x epochs without improvement',
                        default=10, type=int)
    parser.add_argument('--lrd_fac', help='Factor that learning rate is decayed by',
                        default=0.5, type=float)
    parser.add_argument('--es_pat', help='Epochs to wait for an improvement in val_loss before stopping',
                        default=50, type=int)
    opt = parser.parse_args()

    # ----------------
    #  Initialization
    # ----------------
    model = model_init(opt.model_type)                                              # Pre-trained model
    criterion = nn.CrossEntropyLoss()                                               # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)  # Model optimiser
    lr_scheduler = sched_init(optimizer, opt.lrd_epc, opt.lrd_fac)                  # Lr decay scheduler
    data_labels = json.load(open(opt.labels))                                       # Load data labels
    run_dir = generate_id(opt.out_dir, opt.run_id)                                  # Create a unique dir for output
    log_path = "%s/train_log.txt" % run_dir                                         # Save logs to run_dir
    logging.basicConfig(filename=log_path, level=logging.INFO)                      # Initialize logging
    logging.info(opt)                                                               # Record params in log file

    # Assign model to GPU if present:
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        model, criterion, device = gpu_init(model, criterion)
    else:
        device = "cpu"

    # ----------------
    #  Model Training
    # ----------------
    logging.info("[------ Beginning Model Training -------]")

    # Initialise data loaders:
    train_loader, gan_dir = dataloader(opt.src_dir, data_labels['labels'], opt.split_file, 'train',
                                       opt.crop_size, opt.batch_size, opt.workers, opt.gan_params, opt.run_id)
    val_loader, _ = dataloader(opt.src_dir, data_labels['labels'], opt.split_file, 'val',
                               opt.crop_size, opt.batch_size, opt.workers, None, opt.run_id)

    # Run Model training loop:
    train_loop(model, train_loader, val_loader, device, optimizer, criterion, lr_scheduler,
               opt.epochs, opt.es_pat, run_dir)

    # Delete any GAN-generated images if created:
    if opt.gan_params is not None:
        logging.info("Removing GAN Image Directory...")
        rmtree(gan_dir, ignore_errors=True)


if __name__ == "__main__":
    train_dlc()
