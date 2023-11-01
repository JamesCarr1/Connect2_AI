## Imported Modules
 - [torch](https://pytorch.org/docs/stable/index.html)
 - [typing.Tuple]
 - [tqdm.trange]

## Functions

##### `train_step(model, dataloader, value_loss_fn, prior_loss_fn, optimizer, device, value_acc_fn, alpha)`

Performs a single epoch training step.

args:
- `model` - Model to be trained
- `dataloader` - The train dataloader
- `value_loss_fn` - The loss function for the value head
- `prior_loss_fn` - The loss function for the prior head
- `optimizer`
- `device` - Device to compute on
- `value_acc_fn` - The accuracy function used for the value
- `alpha` - Weighting of the value_loss. Total loss is given by:
				`loss = alpha * vloss + (1 - alpha) * ploss*`
returns:
- `value_loss`
- `prior_loss`
- `train_loss`
- `train_acc`

##### `test_step(model, dataloader, value_loss_fn, prior_loss_fn, device, value_acc_fn, alpha)`

Performs a single epoch testing step.

args:
- `model` - Model to be tested
- `dataloader` - The test dataloader
- `value_loss_fn` - The loss function for the value head
- `prior_loss_fn` - The loss function for the prior head
- `device` - Device to compute on
- `value_acc_fn` - The accuracy function used for the value
- `alpha` - Weighting of the value_loss. Total loss is given by:
				`loss = alpha * vloss + (1 - alpha) * ploss*`
returns:
- `value_loss`
- `prior_loss`
- `test_loss`
- `test_acc`

##### `train()`

Combines train_step and test_step to train a model for a number of epochs, storing the evaluation metrics throughout.

args:
- `model` - Model to be trained
- `train_dataloader` - torch.utils.data.DataLoader,
- `test_dataloader` - The test dataloader
- `optimizer`
- `value_loss_fn` - The loss function for the value head
- `prior_loss_fn` - The loss function for the prior head
- `epochs` - Number of training epochs
- `device` - Device to compute on
- `value_acc_fn` - The accuracy function used for the value
- `alpha` - Weighting of the value_loss. Total loss is given by:
				`loss = alpha * vloss + (1 - alpha) * ploss*`
- `generation` - Current parameter generation
- `scheduler` - Optimizer scheduler (optional)
- `print_every` - Number of generations between each print()
- `results` - Dictionary of training and testing loss and accuracy
returns:
- `results` - Dictionary of training and testing loss and accuracy