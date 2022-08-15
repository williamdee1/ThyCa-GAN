import time
import json
import numpy as np
import torch
import glob
import os
import logging
from statistics import mean
from modules.train_eval import model_eval, calc_accuracy
from modules.viz import plot_training_curves


def train_loop(model, train_loader, val_loader, device, optimizer, criterion, lr_scheduler,
               epochs, es_pat, run_dir):
    """
    Perform a full training loop for a model using the dataloader objects and hyper parameters passed.
    """
    # ---------------------
    #  Hold Epoch Results
    # ---------------------
    train_loss_epc = []
    train_acc_epc = []
    val_loss_epc = []
    val_acc_epc = []

    min_val_loss = np.Inf                   # Setting initial validation loss metric to infinity
    train_batches = len(train_loader)       # No. of training batches

    # ---------------------
    #  Begin Training Loop
    # ---------------------
    for epoch in range(epochs):

        model.train()                       # Put model in training mode

        # ---------------------
        #  Hold Batch Results
        # ---------------------
        train_loss_btc = []
        train_acc_btc = []
        val_loss_btc = []
        val_acc_btc = []
        batch_start = time.process_time()

        for images, labels in train_loader:
            # ---------------------
            #  Batch Training
            # ---------------------
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()

            # Predict classes from model inputs:
            mod_preds = model(images)

            # Calculate loss:
            train_loss = criterion(mod_preds, labels)

            # Calculate accuracy of predictions:
            train_acc = calc_accuracy(mod_preds, labels)

            # Calculate gradients and update model:
            train_loss.backward()
            optimizer.step()

            # ---------------------
            #  Batch Eval.
            # ---------------------
            # Put model in evaluation mode:
            model.eval()

            # Turn off gradients for validation:
            with torch.no_grad():
                val_loss, val_acc = model_eval(model, val_loader, criterion, device)

            # ---------------------
            #  Store Batch Results
            # ---------------------
            train_loss_btc.append(train_loss.item())
            train_acc_btc.append(train_acc)
            val_loss_btc.append(val_loss)
            val_acc_btc.append(val_acc)

        # ---------------------
        #  Log Results
        # ---------------------
        # Averaging loss and accuracies across batches:
        av_bch_tl = mean(train_loss_btc)
        av_bch_ta = mean(train_acc_btc)
        av_bch_vl = mean(val_loss_btc)
        av_bch_va = mean(val_acc_btc)

        time_taken = time.process_time() - batch_start

        logging.info(
            "[Epoch %d/%d] [Batches: %d] [Train Loss: %f] [Train Acc: %f] [Val Loss: %f] [Val Acc: %f] [Time Taken: %f]"
            % (epoch + 1, epochs, train_batches, av_bch_tl, av_bch_ta, av_bch_vl, av_bch_va, time_taken)
        )

        # ---------------------
        #  Store Epoch Results
        # ---------------------
        train_loss_epc.append(av_bch_tl)
        train_acc_epc.append(av_bch_ta)
        val_loss_epc.append(av_bch_vl)
        val_acc_epc.append(av_bch_va)

        # ---------------------
        #  Manual Callbacks
        # ---------------------
        # Saving best model (epoch with lowest av. validation loss) throughout training:
        if av_bch_vl <= min_val_loss:
            # Remove previous best model in dir:
            for modpath in glob.glob(os.path.join(run_dir, '*.pth')):
                os.remove(modpath)
            torch.save(model.state_dict(), '%s/vl_%.4f.pth' % (run_dir, av_bch_vl))
            # Updating min_val_loss to new lowest:
            min_val_loss = av_bch_vl

        # Early stopping:
        last_x_epochs = val_loss_epc[-es_pat:]
        if min(last_x_epochs) > min(val_loss_epc):
            logging.info("Training ended. No improvement in val. loss for %d epochs." % es_pat)
            break

        # Update learning rate scheduler (reduces lr if val_loss plateaus):
        lr_scheduler.step(av_bch_vl)

    # ---------------------
    #  Storing Metrics
    # ---------------------
    res_dict = {'train_loss': train_loss_epc, 'train_acc': train_acc_epc,
                'val_loss': val_loss_epc, 'val_acc': val_acc_epc}

    with open('training_results.json', 'w') as f:                       # Save logs to json file
        json.dump(res_dict, f)

    plot_training_curves(res_dict, run_dir)                             # Saving plots of training stats

    logging.info("[------ Training complete. Lowest validation loss of %.4f during training.]" % min(val_loss_epc))

    opt_model_path = '%s/vl_%.4f.pth' % (run_dir, min_val_loss)         # Path to optimal model during training

    return opt_model_path

