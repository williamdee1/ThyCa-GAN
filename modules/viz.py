import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics
sns.set_style("whitegrid")
sns.set(font_scale = 1.0)


def plot_training_curves(res_dict, run_dir):
    """
    Plots training and validation loss and accuracy throughout training.
    """
    train_data = pd.DataFrame.from_dict(res_dict)                   # Convert to pandas dataframe
    train_data.insert(0, 'epoch', range(0, len(train_data)))        # Insert epoch column

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), dpi=100)

    sns.lineplot(data=train_data, x="epoch", y="train_loss", ax=ax1, color='coral', label="Training Loss")
    sns.lineplot(data=train_data, x="epoch", y="val_loss", ax=ax1, color='mediumaquamarine', label="Validation Loss")
    sns.lineplot(data=train_data, x="epoch", y="train_acc", ax=ax2, color='coral', label="Training Accuracy")
    sns.lineplot(data=train_data, x="epoch", y="val_acc", ax=ax2, color='mediumaquamarine', label="Validation Accuracy")

    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax2.set_xlabel("Training Epochs")
    ax2.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0)

    ax2.legend(loc='lower right')
    plt.savefig('%s/Training_Curves.png' % run_dir, bbox_inches='tight')


def conf_mat(preds_df, run_dir):
    # Creating matrices:
    cm_1 = metrics.confusion_matrix(preds_df['class'], preds_df['pred_class'])                          # counts
    cm_2 = metrics.confusion_matrix(preds_df['class'], preds_df['pred_class'], normalize='true')        # percentages

    # Plotting:
    sns.set(font_scale=1.1)
    plt.figure(figsize=(10, 7), dpi=100)

    # Extracting counts and percentages from the matrices above:
    counts = ["{0:,}".format(value) for value in cm_1.flatten()]
    percentages = ["({0:.2%})".format(value) for value in cm_2.flatten()]

    # Totalling the counts of each class:
    nptc_sum = str(cm_1.flatten()[0:2].sum())
    ptc_sum = str(cm_1.flatten()[2:4].sum())
    totals = [nptc_sum, nptc_sum, ptc_sum, ptc_sum]

    # Combining counts, totals and percentages as one label:
    labels = [f"{v1}/{v2}\n{v3}" for v1, v2, v3 in zip(counts, totals, percentages)]

    # Reshaping the labels to fit the array:
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm_2, annot=labels, fmt='', cmap='BuPu', vmin=0, vmax=1,
                xticklabels=['Non-PTC-like', 'PTC-like'], yticklabels=['Non-PTC-like', 'PTC-like'])

    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.savefig('%s/Conf_Matrix.png' % run_dir, bbox_inches='tight')
