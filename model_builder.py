import torch
import data_generator
import connect2
import utils

from torch import nn

class LinearModelV0(torch.nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        # Generation of weights (higher gens have been trained longer)
        self.gen = 1

        self.input_shape = input_shape
        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU()
        )

        # Two types of output: action (i.e choose the best outcome) and value
        self.action_head = nn.Linear(in_features=hidden_units, out_features=input_shape) # outputs a value for ALL board positions, then mask impossible ones
        self.value_head = nn.Linear(in_features=hidden_units, out_features=output_shape)
    
    def __repr__(self):
        return "LinearModelV0"
    
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
    
    def predict_priors(self, board_state: list, possible_moves: list) -> list:
        """
        Runs the model once, and manipulates the results to return just the priors.
        args:
            board_state: the current board state
            possible_moves: the indexes of all LEGAL moves
        returns:
            priors: a list containing the probabilities of choosing each legal move.
        """
        x = torch.tensor(board_state, dtype=torch.float32) # convert board state to a tensor
        possible_moves = torch.tensor(possible_moves, dtype=torch.int64) # possible moves also needs to be converted to a tensors

        # Make a prediction
        self.eval()
        with torch.inference_mode():
            action_ratings, _ = self.forward(x) # run x through the model
        self.train()

        priors = utils.softmax_extract_mean(actions_logits=action_ratings,
                                            legal_moves=possible_moves)

        return priors.tolist() # return just the values

    def predict_value(self, board_state: list) -> float:
        """
        Runs the model once, and manupulates the results to return just the values.
        args:
            board_state: the current board state
        returns:
            value: value from -1 to 1 predicting the result.
        """
        x = torch.tensor(board_state, dtype=torch.float32) # convert to tensor

        # Make a prediction
        self.eval()
        with torch.inference_mode():
            _, value_logits = self.forward(x) # value needs to be normalised to -1 to 1. tanh is perfect for this
        self.train()

        value = torch.tanh(value_logits) # converts to -1 to 1

        return value.item() # returns just the float
    

class EvenModel(torch.nn.Module):
    """
    Returns all possible game states with an even prior
    """
    def __init__(self):
        super().__init__()

    def forward(self, game_state, legal_moves):
        value = 0.0 # Always predicts a draw
        if bool(legal_moves):
            priors = [1/len(legal_moves) for _ in legal_moves]

        return priors, value

class NaiveModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, game_state, legal_moves):
        # Weight internal board positions with twice the weight as external ones
        value = 0
        weights = [0.2, 0.4, 0.4, 0.2]
        for i, piece in enumerate(game_state):
            value += piece * weights[i]
        
        priors = [0] * len(legal_moves) # know the length of the priors vector
        for i, move in enumerate(legal_moves): # cycle through moves and weight central square moves more heavily
            if move == 0 or move == 3:
                priors[i] = 1
            else:
                priors[i] = 2 # weight central squares more highly
        
        # Now normalise priors
        total = sum(priors)
        normalised_priors = [prior / total for prior in priors]

        return normalised_priors, value


if __name__ == '__main__':
    board_state = torch.tensor([1, 0, 0, -1], dtype=torch.float32)
    possible_moves = torch.tensor([1, 2], dtype=torch.int64)
    model = LinearModelV0(4, 3, 1)

    improved_priors = torch.tensor([0.25, 0.75], dtype=torch.float32)

    action_ratings, val = model(board_state)
    
    print(f"Action ratings: {action_ratings} | Value: {val}")

    priors = action_ratings.gather(dim=0, index=possible_moves) # Extract just the relevant priors

    print(f"Relevant action ratings: {priors}")

    priors = torch.softmax(priors, dim=0)

    print(f"Priors: {priors}")

    target_ARs = torch.zeros(4).scatter(dim=0, index=possible_moves, src=improved_priors)

    print(f"Target Priors: {target_ARs}")



