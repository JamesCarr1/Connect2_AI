Base model class including `predict()` and `parallel_predict()` methods.
## Superclass
- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

## Attributes
- `device` - The device the model is currently on. ***NEEDS TO BE REMOVED***
- `gen` - The current generation of the model. Starts at `1`.

## Methods
##### `__init__(device)`

Calls `super().__init__()` and sets the default value of `self.device` to `'cpu'`. Also sets `self.gen` to 1.
##### `to(device)`

Calls `super().to()` and also updates `self.device`
##### `predict(board_state)`

Runs the model with `board_state` and returns the priors (softmaxed) and value (tanhed). To be used in evaluation (.eval() and inference_mode()), not training.

args:
- `board_state` - the board state to be evaluated.
returns:
- `action_probs`: `list[float]` - model prior predictions.
- `value`: `float` - model evaluation.
##### `parallel_predict(board_states)`

Runs the model with multiple board states and predicts priors and values. To be used in evaluation (.eval() and inference_mode()), not training.

args:
- `board_states` - all board states to be evaluated.
returns:
- `action_probs`: `torch.tensor`- model prior predictions.
- `value`: `torch.tensor` - model evaluation.