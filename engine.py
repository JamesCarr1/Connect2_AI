import torch

from typing import Tuple
from tqdm import trange

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               value_loss_fn: torch.nn.Module,
               prior_loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               value_acc_fn,
               alpha) -> Tuple[float, float, float, float]:
    """
    Performs a single epoch training step.

    args:
        model - Model to be trained
        dataloader - The train dataloader
        value_loss_fn - The loss function for the value head
        prior_loss_fn - The loss function for the prior head
        optimizer
        device - Device to compute on
        value_acc_fn - The accuracy function used for the value
        alpha - Weighting of the value_loss. Total loss is given by:
    returns:
        value_loss
        prior_loss
        train_loss
        train_acc

    """
    # Put model in training mode
    model.train()

    train_loss, value_loss, prior_loss, acc = 0, 0, 0, 0

    # Loop through the dataloader batches
    for board_state, legal_moves_mask, expanded_target_priors, winner in dataloader:
        
        board_state, legal_moves_mask = board_state.to(device), legal_moves_mask.to(device)
        expanded_target_priors, winner = expanded_target_priors.to(device), winner.to(device)

        # Unsqueeze winner tensor (to be same size as value_pred)
        winner = winner.unsqueeze(dim=1)

        actions_logits, value_logits = model(board_state)
        value_pred = torch.tanh(value_logits)
        prior_pred = torch.softmax(actions_logits, dim=1)

        ### Calculate loss

        vloss = value_loss_fn(value_pred, winner)
        ploss = prior_loss_fn(actions_logits, expanded_target_priors) # CrossEntropyLoss is between LOGITS and target
        loss = alpha * vloss +  (1 - alpha) * ploss

        value_loss += vloss.item()
        prior_loss += ploss.item()
        train_loss += loss.item()

        ### Zero grad, loss backward and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Calculate accuracy metrics
        acc += value_acc_fn(value_pred.round() + 1, winner + 1).item() # value is -1 to 1 but torchmetrics accuracy wants 0 to 2

    # Adjust metrics to get average loss and accuracy per batch
    value_loss = value_loss / len(dataloader)
    prior_loss = prior_loss / len(dataloader)
    train_loss = train_loss / len(dataloader)
    acc = acc / len(dataloader)

    return value_loss, prior_loss, train_loss, acc

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               value_loss_fn: torch.nn.Module,
               prior_loss_fn: torch.nn.Module,
               device: torch.device,
               value_acc_fn,
               alpha) -> Tuple[float, float]:
    """
    Performs a single epoch testing step.

    args:
        model - Model to be tested
        dataloader - The test dataloader
        value_loss_fn - The loss function for the value head
        prior_loss_fn - The loss function for the prior head
        device - Device to compute on
        value_acc_fn - The accuracy function used for the value
        alpha - Weighting of the value_loss
    returns:
        value_loss
        prior_loss
        test_loss
        test_acc
    """
    # Put model in training mode
    model.eval()

    test_loss, value_loss, prior_loss, acc = 0, 0, 0, 0

    with torch.inference_mode():
        # Loop through the dataloader batches
        for board_state, legal_moves_mask, expanded_target_priors, winner in dataloader:
            
            board_state, legal_moves_mask = board_state.to(device), legal_moves_mask.to(device)
            expanded_target_priors, winner = expanded_target_priors.to(device), winner.to(device)

            # Need to unqueeze winner
            winner = winner.unsqueeze(dim=1)

            actions_logits, value_logits = model(board_state)
            value_pred = torch.tanh(value_logits)
            prior_pred = torch.softmax(actions_logits, dim=1)

            ### Calculate loss

            vloss = value_loss_fn(value_pred, winner)
            ploss = prior_loss_fn(actions_logits, expanded_target_priors) # CrossEntropyLoss is between LOGITS and target
            loss = alpha * vloss +  (1 - alpha) * ploss

            value_loss += vloss.item()
            prior_loss += ploss.item()
            test_loss += loss.item()

            ### Calculate accuracy metrics
            acc += value_acc_fn(value_pred.round() + 1, winner + 1).item() # value is -1 to 1 but torchmetrics accuracy wants 0 to 2

    # Adjust metrics to get average loss and accuracy per batch
    value_loss = value_loss / len(dataloader)
    prior_loss = prior_loss / len(dataloader)
    test_loss = test_loss / len(dataloader)
    acc = acc / len(dataloader)

    return value_loss, prior_loss, test_loss, acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          value_loss_fn: torch.nn.Module,
          prior_loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          value_acc_fn,
          alpha,
          generation,
          scheduler=None,
          print_every=0,
          results=None):
    """
    Combines train_step and test_step to train a model for a number of epochs, storing the evaluation metrics
    throughout.

    args:
        model - Model to be trained
        train_dataloader - torch.utils.data.DataLoader,
        test_dataloader - The test dataloader
        optimizer
        value_loss_fn - The loss function for the value head
        prior_loss_fn - The loss function for the prior head
        epochs - Number of training epochs
        device - Device to compute on
        value_acc_fn - The accuracy function used for the value
        alpha - Weighting of the value_loss
        generation - Current parameter generation
        scheduler - Optimizer scheduler (optional)
        print_every - Number of generations between each print()
        results - Dictionary of training and testing loss and accuracy
    returns:
        results - Dictionary of training and testing loss and accuracy
    """
    if results is None:
        # Create empty results dictionary
        results = {
        "value_train_loss": [],
        "prior_train_loss": [],
        "total_train_loss": [],
        "train_acc": [],
        "value_test_loss": [],
        "prior_test_loss": [],
        "total_test_loss": [],
        "test_acc": []
        }

    # Loop through training and testing steps
    for epoch in trange(epochs, desc="Epoch", position=1, leave=False):
        # Train
        value_train_loss, prior_train_loss, total_train_loss, train_acc = train_step(model=model,
                                                                                    dataloader=train_dataloader,
                                                                                    value_loss_fn=value_loss_fn,
                                                                                    prior_loss_fn=prior_loss_fn,
                                                                                    optimizer=optimizer,
                                                                                    device=device,
                                                                                    value_acc_fn=value_acc_fn,
                                                                                    alpha=alpha)
        # Test
        value_test_loss, prior_test_loss, total_test_loss, test_acc = test_step(model=model,
                                                                                dataloader=test_dataloader,
                                                                                value_loss_fn=value_loss_fn,
                                                                                prior_loss_fn=prior_loss_fn,
                                                                                device=device,
                                                                                value_acc_fn=value_acc_fn,
                                                                                alpha=alpha)
        
        if scheduler is not None:
            scheduler.step()

        if print_every and (generation % print_every == 0):
            print(
            f"Epoch: {epoch+1} | "
            f"total_train_loss: {total_train_loss:.5f} | "
            f"train_acc: {train_acc:.5f} | "
            f"total_test_loss: {total_test_loss:.5f} | "
            f"test_acc: {test_acc:.5f}"
            )

        # Update results dict
        results["value_train_loss"].append(value_train_loss)
        results["prior_train_loss"].append(prior_train_loss)
        results["total_train_loss"].append(total_train_loss)
        results["train_acc"].append(train_acc)

        results["value_test_loss"].append(value_test_loss)
        results["prior_test_loss"].append(prior_test_loss)
        results["total_test_loss"].append(total_test_loss)
        results["test_acc"].append(test_acc)

    return results