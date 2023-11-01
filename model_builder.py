import torch
from torch import nn

class Model(nn.Module):
    """
    Class (interface) which contains .predict() and .parallel_predict() methods which are used by all models.
    """
    def __init__(self):
        super().__init__()
        self.device = 'cpu' # models are always instantiated on cpu
        self.gen = 1
    
    def to(self, device):
        """
        Sends model to device and updates self.device. PROBABLY SHOULDN'T BE USED
        """
        super().to(device)
        self.device = device

        return self
    
    def predict(self, board_state):
        """
        Runs the model once and returns the priors (softmaxed) and value (tanhed)
        """
        # Tensor must be transferred to self.device (as they are generated from lists, so are 'cpu' by default)
        # x must also be unsqueezed, as it is to be passed through a model which needs batch_size as first dimension
        x = torch.tensor(board_state, dtype=torch.float32).unsqueeze(dim=0).to(self.device) # convert to tensor

        # Make a prediction
        self.eval()
        with torch.inference_mode():
            action_logits, value_logits = self.forward(x) # value needs to be normalised to -1 to 1. tanh is perfect for this
        self.train()

        # Apply activation functions. Apply to dim=1 as it has not yet been squeezed.
        action_probs = torch.softmax(action_logits, dim=1).squeeze() # need to squeeze back after unsqueeze
        value = torch.tanh(value_logits).squeeze()

        return action_probs.tolist(), value.item()

    def parallel_predict(self, board_states):
        """
        Runs the model with multiple board states and predicts priors and values. Note eval mode is NOT called here
        """
        # Send to device
        x = board_states.to(self.device)

        with torch.inference_mode():
            action_logits, value_logits = self.forward(x)
        
        # Apply activation functions
        action_probs = torch.softmax(action_logits, dim=1)
        value = torch.tanh(value_logits)

        return action_probs, value
        
class ConvModelV0(Model):
    """
    NN model with convolutional layers.
    """
    def __init__(self, input_shape, hidden_units, output_shape, kernel_size):
        super().__init__()
        self.input_shape = input_shape

        # Convolutional layer
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_units, kernel_size=kernel_size)
        )

        # Linear layer
        self.layer_1 = nn.Sequential(
            nn.Flatten(), # flatten back to (batch_size, units) from (batch_size, 1, units)
            nn.Linear(in_features=hidden_units * (5 - kernel_size), out_features=hidden_units)
        )

        # Two types of output: action (i.e choose the best outcome) and value
        self.action_head = nn.Linear(in_features=hidden_units, out_features=input_shape) # outputs a value for ALL board positions, then mask impossible ones
        self.value_head = nn.Linear(in_features=hidden_units, out_features=output_shape)
    
    def __repr__(self):
        return "ConvModelV0"

    def forward(self, x):
        """
        args:
            x: current board position
        returns:
            action_rating: tensor of length 4, contains a rating for ALL POSSIBLE MOVES
            value: model rating of current moves
        """
        # Need to unsqueeze dim for conv1d
        # Pass through first layer
        x = self.layer_1(self.conv_layer(x.unsqueeze(dim=1)))

        # Now pass through specialised heads
        action_logits = self.action_head(x)
        value_logits = self.value_head(x) # NOTE THESE ARE NOT NORMALISED TO -1 < y < 1

        return action_logits, value_logits
    
class NonLinearModelV0(Model):
    """
    NN model with non-linear layers.
    """
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.input_shape = input_shape

        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU()
        )

        # Two types of output: action (i.e choose the best outcome) and value
        self.action_head = nn.Linear(in_features=hidden_units, out_features=input_shape) # outputs a value for ALL board positions, then mask impossible ones
        self.value_head = nn.Linear(in_features=hidden_units, out_features=output_shape)
    
    def __repr__(self):
        return "NonLinearModelV0"

    def forward(self, x):
        """
        args:
            x: current board position
        returns:
            action_rating: tensor of length 4, contains a rating for ALL POSSIBLE MOVES
            value: model rating of current moves
        """
        # Pass through first layer
        x = self.layer_1(x)

        # Now pass through specialised heads
        action_rating = self.action_head(x)
        value_logits = self.value_head(x) # NOTE THESE ARE NOT NORMALISED TO -1 < y < 1
        
        return action_rating, value_logits

class EvenModel(Model):
    """
    Returns all possible game states with an even prior
    """
    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return "EvenModel"

    def forward(self, x):
        value = 0.0 # Always predicts a draw
        priors = [1 / len(x) for _ in x] # Predict even priors

        return priors, value

class NaiveModel(Model):

    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return "NaiveModel"

    def forward(self, x):
        # Weight internal board positions with twice the weight as external ones
        value = 0
        weights = [0.2, 0.3, 0.3, 0.2]
        for i, piece in enumerate(x):
            value += piece * weights[i]

        return weights, value

class NaiveUnevenModel(Model):

    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return "NaiveUnevenModel"

    def forward(self, x):
        # Weight internal board positions with twice the weight as external ones
        value = 0
        weights = [0.21, 0.29, 0.31, 0.19]
        for i, piece in enumerate(x):
            value += piece * weights[i]

        return weights, value

if __name__ == '__main__':
    board_state = torch.tensor([1, 0, 0, -1], dtype=torch.float32)
    possible_moves = torch.tensor([1, 2], dtype=torch.int64)
    model = NonLinearModelV0(4, 3, 1)

    improved_priors = torch.tensor([0.25, 0.75], dtype=torch.float32)

    action_ratings, val = model(board_state)
    
    print(f"Action ratings: {action_ratings} | Value: {val}")

    priors = action_ratings.gather(dim=0, index=possible_moves) # Extract just the relevant priors

    print(f"Relevant action ratings: {priors}")

    priors = torch.softmax(priors, dim=0)

    print(f"Priors: {priors}")

    target_ARs = torch.zeros(4).scatter(dim=0, index=possible_moves, src=improved_priors)

    print(f"Target Priors: {target_ARs}")

if __name__ == '__main__':
    model = ConvModelV0(input_shape=4,
                        hidden_units=16,
                        output_shape=1,
                        kernel_size=3).to('cpu')
    
    for i in range(10000):
        model.predict([1, 0, 0, 1])

