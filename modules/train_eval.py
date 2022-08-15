import torch
from statistics import mean


def model_eval(model, dataloader, criterion, device):
    """
    Evaluates the model using either a test or validation set.
    Returns model loss and accuracy averaged over batches.
    """

    test_loss = []
    accuracy = []

    for images, labels in dataloader:

        # Assigning to GPU/CPU device:
        images, labels = images.to(device), labels.type(torch.LongTensor).to(device)

        # Getting model predictions and associated loss on test set data:
        mod_preds = model(images)
        loss = criterion(mod_preds, labels)
        test_loss.append(loss.item())

        # Calculate accuracy of predictions:
        acc = calc_accuracy(mod_preds, labels)
        accuracy.append(acc)

    # Return average loss and accuracy metrics across batches:
    av_test_loss = mean(test_loss)
    av_accuracy = mean(accuracy)

    return av_test_loss, av_accuracy


def calc_accuracy(mod_preds, labels):
    """ Calculates accuracy of model predictions """
    # Convert un-normalized scores to probabilities via softmax:
    probs = torch.nn.functional.softmax(mod_preds, dim=1)

    # Get tensors for the predicted label and probability for that label for each image:
    # I.e. probs = [0.6, 0.4], then -> top_p = 0.6, top_label = 0.0
    top_p, top_label = probs.topk(1, dim=1)
    # Boolean check if top_label is the correct label:
    bool_check = top_label == labels.view(*top_label.shape)
    # Calculate accuracy based on correct labels:
    acc = torch.mean(bool_check.type(torch.FloatTensor)).item()

    return acc


