import torch

def softmax_extract_mean(actions_logits: torch.Tensor,
                         legal_moves: torch.Tensor):
    """
    Applies softmax function to logits. Then extracts the legal moves and normalises the result.
    """
    priors = actions_logits.softmax(dim=0) # apply softmax
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
    pred_score = torch.round(preds) # rounds to nearest integer to output a score
    acc = sum([torch.equal(pred_score, labels[i]) for i, pred_score in enumerate(pred_score)]) / preds.shape[0]

    return acc

if __name__ == '__main__':
    priors = torch.tensor([[1, 1],
                            [2, 2],
                            [3, 3]], dtype=torch.float32)
    
    mask = torch.tensor([[1, 1],
                        [1, 0],
                        [1, 1]], dtype=torch.float32)
    
    print(priors.shape)

    print(softmax_mask_mean(priors, mask))