NN model with non-linear layers.
## Superclass
[[Model]]

## Attributes
- `input_shape`: `int` - Shape of game board (4 for connect2).
- `hidden_units`: `int` - Number of hidden units in each hidden layer.
- `output_shape`: `int` - Output shape of value head (normally 1).

- `layer_1` - Non-linear layer
- `action_head` - Layer creating prior outputs.
- `value_head` - Layer creating value output.

## Methods
##### `forward(x)`

Performs a forward pass on input data `x`.

args:
- `x` - Data to perform forward pass on.
returns:
- `action_logits` - Prior output.
- `value_logits` - Evaluation output.