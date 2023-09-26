import torch
import math
import random
import scipy.stats
import os
import time

import pandas as pd

import connect2
import model_builder
import utils

from matplotlib import pyplot as plt
from pathlib import Path

class Node:
    def __init__(self, prior, to_play):
        self.prior = prior # The probability of selecting this state from it's parents
        self.to_play = to_play # The player whose turn it is (-1 or 1)

        self.children = {}
        self.visit_count = 0 # Number of times the state has been visited
        self.value_sum = 0 # The total value of this state from all visits
        self.outcome = None # The outcome at this node (1, 0 or -1 for the three outcomes). None for unfinished game
        self.depth = 0
        self.game_state = None # board position at this node

    def value(self):
        # Returns an averaged value
        if self.visit_count > 0:
            return self.value_sum / self.visit_count
        else:
            return 0
    
    def expand(self, model: torch.nn.Module, game: connect2.Connect2Game,
               state):
        """
        Expands the tree with all possible moves from the current state.
        args:
            model: torch.nn.Module: model that provides the prior value
            game: connect2.Connect2Game: object containing game data.
            board_position: the arrangement of pieces on the board at this nodes
        """
        # Update the game with the current position
        game.game_state = state
        game.update_available_moves()

        # Update this node's board position
        self.game_state = state.copy()

        priors = model.predict_priors(self.game_state, game.available_moves)
        for move, prior in zip(game.available_moves, priors):
            self.children[move] = Node(prior, self.to_play * -1) # Add the node to the tree
            self.children[move].find_legal_moves(game)
            self.children[move].set_depth(self.depth + 1)

    def expanded(self):
        """
        Checks if node has been expanded i.e if children has elements
        """
        return len(self.children) != 0 # if there ARE elements in children, returns true

    def set_depth(self, depth):
        self.depth = depth
    
    def find_legal_moves(self, game: connect2.Connect2Game):
        """
        Updates legal moves and stores them
        """
        game.set_game_state(self.game_state)
        self.legal_moves = game.available_moves

    def select_child(self):
        """
        LEGACY: Randomly selects a child based on their ucb score
        """
        # ucb_score outputs values >= -1. Sum of weights must be >= -1 for random.choices, so add 1 to all values.
        weights = [x + 1 for x in map(self.ucb_score, self.children.values())] 
        if weights == [0] * len(weights):
            weights = [1] * len(weights) # sometimes weights are all zeros. Adjust this
        
        return random.choices(list(self.children.items()), weights)[0] # randomly choose based on weights.
    
    def select_child(self):
        """
        Randomly selects a child based on their ucb score
        """
        # ucb score can output scores >= -1. THerefore, use a normal curve to convert negative scores to +ve probs
        scores = list(map(self.ucb_score, self.children.values()))# have to extract just the value for weights
        maximum_score = max(scores) # find the maximum to use as a mean (therefore max_score will have max prob)

        weights = scipy.stats.norm(maximum_score, 1).pdf(scores) # find weights from normal curve. COULD ADJUST STD DEV???

        return random.choices(list(self.children.items()), weights)[0] # randomly choose based on weights.

    def select_action(self, choose_type):
        """
        Given a Monte Carlo searched tree, chooses the move with the highest value.
        
        args:
            type: 'greedy' or 'random'. 'greedy' selects the action with the highest value. 'random' weights the actions and chooses one
        """
        priors = [node.prior for node in self.children.values()]
        if choose_type == 'greedy':
            values = [node.value() for node in self.children.values()] # find all values (these are improved prior logits)
            improved_priors = torch.tensor(values).softmax(dim=0).tolist() # softmax to get probabilities

            # returns the child with the highest value
            return priors, improved_priors, max(self.children.items(), key=lambda x: x[1].value()), list(self.children.keys())
        
        elif choose_type == 'random':
            values = [node.value() for node in self.children.values()] # find all values (these are improved prior logits)
            improved_priors = torch.tensor(values).softmax(dim=0).tolist() # softmax to get probabilities

            # Recompute values into weights
            weights = scipy.stats.norm(max(values), 1).pdf(values)

             # randomly choose from weights. Also return priors, MCTSed priors and legal moves
            return priors, improved_priors, random.choices(list(self.children.items()), weights)[0][0], list(self.children.keys())

    def ucb_score(self, child):
        """
        Implementation of ucb score that only takes in one value - the child (parent is self)
        """
        return ucb_score(self, child)

    def plot(self, model, game: connect2.Connect2Game):
        """
        Plots the current tree structure extending from this node
        """
        game.set_game_state(self.game_state)
        print(f"{'|---' * self.depth}{self.game_state}: prior: {self.prior}, value: {self.value()}, visit_count: {self.visit_count}")
        for child in self.children.values():
            child.plot(model, game)

class MCTS:
    """
    Implements Monte Carlo tree search
    """
    def __init__(self, model: torch.nn.Module, game=connect2.Connect2Game()):
        self.game = game
        self.model = model
    
    def run(self, state, to_play, num_simulations=5, choose_type='random'):
        root = Node(0, to_play) # Set up root node

        # Expand root
        root.expand(self.model, self.game, state)
        #root.plot(self.model, self.game)

        # Now simulate num_simulation times
        for _ in range(num_simulations):
            node = root # always start from the root
            search_path = [node] # contains the nodes searched in this simulation

            while node.expanded(): # get to an unexpanded node
                action, node = node.select_child() # action: the move required to get from the original node to the new node
                search_path.append(node)

            parent = search_path[-2] # save parent of node
            state = parent.game_state

            next_state = self.game.get_next_state(state, to_play=1, action=action)
            next_state = self.game.reverse_board_view(next_state)
            value = self.game.get_outcome_for_player(next_state, player=1)

            if value is None:
                node.expand(self.model, self.game, next_state)
                value = self.model.predict_value(next_state) # value prediction from model

            
            self.backpropogate(search_path, value, parent.to_play * -1)
            #root.plot(self.model, self.game)
        
        # Can uncomment to plot final tree
        #root.plot(self.model, self.game)

        return root.select_action(choose_type=choose_type) # returns the improved priors and the chosen action

    def backpropogate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

class GameGenerator:
    def __init__(self, model, game_type: connect2.Connect2Game):
        self.game = game_type()
        self.model = model
        self.mcts = MCTS(model=self.model, game=self.game)
    
    def generate_game(self, num_simulations=8):
        # Define starting position
        starting_state = [0, 0, 0, 0]
        self.game.set_game_state(starting_state)
        result = None # game is not over

        # Define lists of positions
        board_states = [starting_state]
        actions = [] # actions that have been taken in the game
        priors = [] # priors output by the model
        improved_priors = [] # values output by the MCTS
        legal_moves = []

        while result is None:
            # Find the model priors, MCTS priors and the chosen action
            ps, vs, action, lms = self.mcts.run(board_states[-1], 1, num_simulations=num_simulations)
            # And save
            priors.append(ps)
            improved_priors.append(vs)
            actions.append(action)
            legal_moves.append(lms)

            # Now make the action
            next_state = self.game.get_next_state(board_states[-1], 1, action)
            next_state = self.game.reverse_board_view(next_state) # reverse the view
            # and save
            board_states.append(next_state)
            # Now check if the game is over
            result = self.game.get_outcome_for_player(next_state, 1)
        
        # Now create winner vector
        winner = [0] * len(board_states)
        for i in range(len(board_states)):
            if i % 2 == 0: # i.e if i is odd, same as result
                winner[i] = result
            else:
                winner[i] = -1 * result
        
        # Can now return all info. Info that is saved: 
        return board_states, winner, actions, priors, improved_priors, legal_moves

    def generate_n_games(self, num_simulations, num_games, save_folder=Path(os.getcwd()) / "generated_games"):
        """
        Generates n games, each with a MCTS of depth num_simulations. Saves the results to save_file.
        """
        # Setup empty df
        df = pd.DataFrame(columns=["Board State", "Winner", "Improved Priors", "Legal Moves"])
        for i in range(num_games):
            # Generate a game
            board_states, winners, _, _, improved_priors, legal_moves = self.generate_game(num_simulations=num_simulations)
            
            # Add the data to the ROW BY ROW
            for i, improved_prior in enumerate(improved_priors):
                # And append to the df
                df.loc[len(df.index)] = [board_states[i], winners[i], improved_prior, legal_moves[i]]
        
        # Check the save folder exists. If not, make it
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        
        save_path = save_folder / f"{self.model}.{self.model.gen}_{num_games}_games_{num_simulations}_MCTS_sims.pkl"

        # Save as pickle to maintain lists as lists (csv converts them to strings)
        df.to_pickle(save_path)


def ucb_score(parent: Node, child: Node):
    """Calculates 'ucb score', which can be used as weights for the MCTS sampling"""
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the opposing player
        child_score = -child.value()
    else:
        child_score = 0
    
    return prior_score + child_score

def normalised_ucb_score(parent: Node, child: Node, Z):
    """Calculates 'ucb_score', then normalises it between 0 """

if __name__ == '__main__':
    model = model_builder.LinearModelV0(input_shape=4, hidden_units=2, output_shape=1)
    game_generator = GameGenerator(model=model, game_type=connect2.Connect2Game)

    num_games = 1000
    num_sims = 15

    start = time.time()
    time_taken = game_generator.generate_n_games(num_games=num_games, num_simulations=num_sims)
    end = time.time()
    print(f"Generated and saved {num_games} games in {end - start} seconds")

    """
    for i, action in enumerate(actions):
        print(f"(Board State, winner): {board_states[i]} | Chosen action: {action} | Priors: {priors[i]} | Improved Priors: {improved_priors[i]} | Legal moves: {legal_moves[i]}")

    print(f"Final position: {board_states[-1]}")
    """

    df2 = pd.read_pickle(Path(os.getcwd()) / "generated_games" / f"LinearModelV0.1_{num_games}_games_{num_sims}_MCTS_sims.pkl")

    print(df2.head(5))
    