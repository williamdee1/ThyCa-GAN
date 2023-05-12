import torch.nn as nn
import torch
import pandas as pd
from modules.viz import conf_mat
import logging
import sklearn
from sklearn.metrics import f1_score


def test_eval(model, mdl_weights_loc, dataloader, device, run_dir):
    # Load the saved weights of the optimal model during training:
    try:
        model.load_state_dict(torch.load(mdl_weights_loc))
        logging.info("Model weights loaded successfully.")
    except RuntimeError:
        # Caused by loading weights from a model that was trained with nn.DataParallel (multiple GPUs)
        dev_ids = list(range(0, torch.cuda.device_count()))                     # Count the no. GPUs
        model = nn.DataParallel(model, device_ids=dev_ids)                      # Parallelize the model
        model.load_state_dict(torch.load(mdl_weights_loc))
        logging.info("Model weights loaded successfully in parallel.")
    else:
        logging.info('Please use >=1 GPU for model evaluation.')

    # Put model in evaluation mode:
    model.eval()

    # Create a dataframe of the model predictions:
    pred_df = test_preds(model, dataloader, device, run_dir)

    # Calculate and display accuracy on a patient and diagnosis level:
    pat_acc, pat_corr, pat_len, diag_accs, num_corr, len_data = grouped_acc(pred_df, run_dir)

    return pat_acc, pat_corr, pat_len, diag_accs, num_corr, len_data


def test_preds(model, dataloader, device, run_dir):
    """
    Evaluates the model using a test set.
    Returns a dataframe containing prediction confidence per image.
    """
    # Lists to hold batch data and predictions:
    patients = []
    diags = []
    act_class = []
    pred_class = []
    pred_prob = []

    for images, ptc_class, patient, diag in dataloader:
        # Assigning to GPU/CPU device:
        images = images.to(device)

        # Getting model predictions on test set data:
        mod_preds = model(images)

        # Convert un-normalized scores to probabilities via softmax:
        probs = torch.nn.functional.softmax(mod_preds, dim=1)

        # Get prediction confidence and prediction label:
        pred_p, pred_label = probs.topk(1, dim=1)

        # ----------------------------
        #  Store Batch Prediction Data
        # ----------------------------
        [patients.append(i) for i in patient]
        [diags.append(i) for i in diag]
        [act_class.append(i) for i in ptc_class.numpy()]
        [pred_class.append(i[0]) for i in pred_label.detach().cpu().numpy()]
        [pred_prob.append(i[0]) for i in pred_p.detach().cpu().numpy()]

    # ---------------------
    #  Storing Predictions
    # ---------------------

    pred_dict = {'patient': patients, 'diagnosis': diags, 'class': act_class,
                 'pred_class': pred_class, 'pred_prob': pred_prob}


    # Save preds as csv file:
    df = pd.DataFrame(pred_dict).sort_values(by='patient')
    df.to_csv("%s/test_preds.csv" % run_dir, index=None)

    return df


def grouped_acc(df, run_dir):
    """
    Determines accuracy and f1 score of classifier on a patient and diagnosis basis.
    Rather than on a per-image basis (as multiple slides exist per patient).
    Designates prediction by majority voting.
    """
    # Group by patient and average actual and predicted class:
    maj_vote_df = df.groupby(['patient', 'diagnosis'])[['class',
                                                        'pred_class']].mean().reset_index()
    # Rounding the prediction to either 0 or 1 (0.5 rounds to 0):
    maj_vote_df['pred_class'] = maj_vote_df['pred_class'].round().to_list()

    # Calculate f1 score:
    f1 = f1_score(maj_vote_df['class'].values, maj_vote_df['pred_class'].values, average='macro')
    logging.info("Patient-level f1 score: %.2f%%" % (f1 * 100))

    # Save confusion matrix:
    conf_mat(maj_vote_df, run_dir)

    # Calculate classifier accuracy on per patient basis:
    pat_str, pat_acc, pat_corr, pat_len = subset_acc(maj_vote_df)
    logging.info("Patient-level accuracy: %s" % pat_str)

    # List of the unique diagnoses present in the data:
    diag_list = maj_vote_df['diagnosis'].unique().tolist()
    diag_accs = []
    num_corr = []
    len_data = []

    for i in range(len(diag_list)):
        diag_df = maj_vote_df[maj_vote_df['diagnosis'] == diag_list[i]
                              ].reset_index(drop=True)
        diag_str, diag_acc, diag_corr, diag_len = subset_acc(diag_df)
        diag_accs.append(diag_acc)
        num_corr.append(diag_corr)
        len_data.append(diag_len)
        logging.info("%s Accuracy: %s" % (diag_list[i], diag_str))

    return pat_acc, pat_corr, pat_len, diag_accs, num_corr, len_data


def subset_acc(subset_df):
    """
    Calculates prediction accuracy for a passed subset dataframe.
    """
    correct = 0
    for i in range(len(subset_df)):
        if subset_df['class'][i] == subset_df['pred_class'][i]:
            correct +=1
        else:
          pass

    len_data = len(subset_df)
    acc = correct/len_data

    # Convert to string representation:
    acc_str = "%.2f%% (%s/%s)" % ((acc*100), correct, len_data)

    return acc_str, acc, correct, len_data


