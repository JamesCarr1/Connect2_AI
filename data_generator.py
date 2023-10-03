import mcts
import connect2
import model_builder
import torch

import os
import time

import pandas as pd

from pathlib import Path

class GameGenerator:
    def __init__(self, model, game_type: connect2.Connect2Game):
        self.game = game_type()
        self.model = model
        self.mcts = mcts.MCTS(model=self.model, game=self.game)
    
    def generate_game(self, num_simulations=8):
        # Define starting position
        starting_state = [0, 0, 0, 0]
        self.game.set_game_state(starting_state)
        result = None # game is not over

        # Define lists of positions
        board_states = [starting_state]
        actions = [] # actions that have been taken in the game
        improved_priors = [] # values output by the MCTS
        legal_moves = []

        while result is None:
            # Run a Monte Carlo tree search
            mcts_root = self.mcts.run(board_states[-1], to_play=1, num_simulations=num_simulations)
            
            # Find the probabilities of each action
            improved_action_probs = [0] * 4
            for move, node in mcts_root.children.items(): # cycle through possible moves
                improved_action_probs[move] = node.visit_count # append the visits
            print(improved_action_probs)
            total = sum(improved_action_probs)
            improved_action_probs = [prob / total for prob in improved_action_probs] # renormalise
            print(improved_action_probs)

            available_moves = [key for key in mcts_root.children.keys()] # get all possible moves
            
            # Choose action (randomly)
            action = mcts_root.select_action(choose_type='random')

            # And record
            improved_priors.append(improved_action_probs)
            actions.append(action)
            legal_moves.append(available_moves)

            # Now make the action and reverse the view
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
        return board_states, winner, actions, improved_priors, legal_moves

    def generate_n_games(self, num_simulations, num_games, save_folder=Path(os.getcwd()) / "generated_games"):
        """
        Generates n games, each with a MCTS of depth num_simulations. Saves the results to save_file.
        """
        # Setup empty df
        df = pd.DataFrame(columns=["Board State", "Winner", "Improved Priors", "Legal Moves"])
        for i in range(num_games):
            # Generate a game
            board_states, winners, _, improved_priors, legal_moves = self.generate_game(num_simulations=num_simulations)
            
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

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_builder.LinearModelV0(input_shape=4, hidden_units=2, output_shape=1).to(device)
    model = model_builder.NaiveUnevenModel2().to(device)
    game_generator = GameGenerator(model=model, game_type=connect2.Connect2Game)

    num_games = 2
    num_sims = 500

    start = time.time()
    time_taken = game_generator.generate_n_games(num_games=num_games, num_simulations=num_sims)
    end = time.time()
    print(f"Generated and saved {num_games} games in {end - start} seconds")

    """
    for i, action in enumerate(actions):
        print(f"(Board State, winner): {board_states[i]} | Chosen action: {action} | Priors: {priors[i]} | Improved Priors: {improved_priors[i]} | Legal moves: {legal_moves[i]}")

    print(f"Final position: {board_states[-1]}")
    """

    df2 = pd.read_pickle(Path(os.getcwd()) / "generated_games" / f"{model}.1_{num_games}_games_{num_sims}_MCTS_sims.pkl")

    print(df2.head(20))
    