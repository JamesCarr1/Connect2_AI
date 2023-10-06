import torch

from matplotlib import pyplot as plt

def softmax_extract_mean(actions_logits: torch.Tensor,
                         legal_moves: torch.Tensor,
                         eps=1e-12):
    """
    Applies softmax function to logits. Then extracts the legal moves and normalises the result.
    """
    priors = actions_logits.softmax(dim=0) + eps # apply softmax. Add eps to prevent divide-by-zero errors
    priors = torch.gather(priors, dim=0, index=legal_moves) # extract just the legal moves
    priors = torch.div(priors, priors.sum()) # renormalise

    return priors

def softmax_mask_mean(actions_logits: torch.Tensor,
                      legal_moves_mask: torch.Tensor):
    """
    Applies softmax function to logits. Then masks out the illegal moves and normalises the result.
    """
    priors = actions_logits.softmax(dim=1) # apply softmax
    priors = torch.mul(priors, legal_moves_mask) # mask out illegal moves
    priors = priors.div(priors.sum(dim=1).unsqueeze(dim=1)) # divide each row by the sum of the row

    return priors


def value_acc(preds, labels):
    """
    Calculates the accuracy between pred and label, given by count(round(pred) == label) / len(pred)
    """
    pred_scores = torch.round(preds) # rounds to nearest integer to output a score
    acc = sum([torch.equal(pred_score, label) for pred_score, label in zip(pred_scores, labels)]) / preds.shape[0]

    return acc

def plot_loss_curves(results, labels):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
                "train_acc": [...],
                "test_loss": [...],
                "test_acc": [...]}
        labels: the labels of the dictionary to plot
    """

    # Figure out how many epochs
    epochs = range(len(results["total_train_loss"]))

    # Plot loss
    plt.subplot(1, 2, 1)
    for label, colour, style in labels:
        plt.plot(epochs, results[label], label=label, c=colour, linestyle=style)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label=f"train_acc", c=colour, linestyle='-')
    plt.plot(epochs, results["test_acc"], label=f"test_acc", c=colour, linestyle='--')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


if __name__ == '__main__':
    priors = torch.tensor([[1, 1],
                            [2, 2],
                            [3, 3]], dtype=torch.float32)
    
    mask = torch.tensor([[1, 1],
                        [1, 0],
                        [1, 1]], dtype=torch.float32)
    
    print(priors.shape)

    print(softmax_mask_mean(priors, mask))