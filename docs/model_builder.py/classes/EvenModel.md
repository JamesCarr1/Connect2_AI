Returns equal probability for each move and always predicts a draw (`0.0`).
## Superclass
[[Model]]
## Methods
##### `forward(x)`

Performs a forward pass on input data `x`.

args:
- `x` - Data to perform forward pass on.
returns:
- `action_logits` - Prior output.
- `value_logits` - Evaluation output.