import torch

def softmax_actions_logits(actions_logits: torch.Tensor,
                           expanded_legal_moves: torch.Tensor):
    """
    Obtains a length 4 actions_logits tensor. Extracts the action rating of the legal moves, softmaxes them and redistributes the resulting
    prior prediction.

    args:
        actions_logits: logits output by the model
        expanded_legal_moves: Contains [legal_moves, [0] * 4 - len(legal_moves), [len(legal_moves)]]
    """
    # Extract the data from expanded_legal_moves
    legal_moves_true_length = int(expanded_legal_moves[-1].item())
    legal_moves = expanded_legal_moves[:legal_moves_true_length].to(torch.int64)

    # Now softmax the relevant logits
    preds = actions_logits.gather(dim=0, index=legal_moves) # extract just the relevant logits
    preds = preds.softmax(dim=0) # softmax
    preds = torch.zeros(4).scatter(dim=0, index=legal_moves, src=preds)

    return preds

def value_acc(preds, labels):
    """
    Calculates the accuracy between pred and label, given by count(round(pred) == label) / len(pred)
    """
    pred_score = torch.round(preds) # rounds to nearest integer to output a score
    acc = sum([torch.equal(pred_score, labels[i]) for i, pred_score in enumerate(pred_score)]) / preds.shape[0]

    return acc

if __name__ == '__main__':
    preds = torch.tensor([0.9, 0.7, -0.2, -0.5, -1])
    labels = torch.tensor([1, 1, 0, 0, -1])

    print(value_acc(preds, labels))