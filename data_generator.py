import mcts
import connect2
import model_builder
import torch

import os
import time

import pandas as pd

from pathlib import Path

class GameInfo:
    """
    Contains information used to generate games.

    attr:
        tree: mcts.Node: the current tree search
        board_states: the positions the board has been in so far
        next_state: the next state to be observed
        actions: the moves made in the game so far
        improved_priors: priors output by the MCTS
        legal_moves: the moves that can be made at each step
        outcome: whether the game has finished
        winner: list of who won from current player's perspective
    """
    def __init__(self,
                 state=(0, 0, 0, 0),
                 tree=None,
                 outcome=None,
                 winner=None):
        self.state = state
        self.tree = tree

        self.board_states = [self.state]
        self.actions = []
        self.improved_priors = []
        self.legal_moves = []

        self.outcome = None
        self.winner = None
    
    def update_improved_priors(self):
        """
        Using the data from self.tree, append the new action probabilities to self.improved_priors
        """
        # Find the probabilities of each action
        action_probs = [0] * 4
        for move, node in self.tree.children.items(): # cycle through possible moves
            action_probs[move] = node.visit_count # append the visits
        total = sum(action_probs)
        action_probs = [prob / total for prob in action_probs] # renormalise
        self.improved_priors.append(action_probs)
    
    def update_legal_moves(self):
        """
        Using data from self.tree, append the list of legal moves to self.legal_moves
        """
        self.legal_moves.append([key for key in self.tree.children.keys()]) # get all possible moves)
    
    def select_action(self):
        """
        Performs mcts.Node.select_action on self.tree. Appends the result to self.actions and returns it
        """
        action = self.tree.select_action(choose_type='random')
        self.actions.append(action)
        return action

    def get_next_state(self, action, game):
        """
        Makes the action on self.state, then reverses the board view. Saves the new state.
        """
        next_state = game.get_next_state(self.state, to_play=1, action=action)
        next_state = game.reverse_board_view(next_state)
        self.board_states.append(next_state)
        self.state = next_state
    
    def update_outcome(self, game):
        """
        Checks if the game is over
        """
        self.outcome = game.get_outcome_for_player(self.state, 1)
    
    def update_winner(self):
        """
        Updates winner vector
        """
        self.winner = [0] * len(self.board_states)
        for i in range(len(self.board_states)):
            if i % 2 == 0: # i.e if i is odd, same as result
                self.winner[i] = self.outcome
            else:
                self.winner[i] = -1 * self.outcome
        
        return self.winner



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
            total = sum(improved_action_probs)
            improved_action_probs = [prob / total for prob in improved_action_probs] # renormalise

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
        # Setup empty lists
        board_states, winners, improved_priors, legal_moves = [], [], [], []
        df = pd.DataFrame(columns=["Board State", "Winner", "Improved Priors", "Legal Moves"])
        for i in range(num_games):
            # Generate a game
            game_data = self.generate_game(num_simulations=num_simulations)

            # Update all the lists
            board_states += game_data[0][:-1] # game_data[0] INCLUDES the last state (final state, outcome is not None), which we don't want
            winners += game_data[1][:-1] # game_data[1] INCLUDES the final result (final state, outcome is not None), which we don't want
            improved_priors += game_data[3]
            legal_moves += game_data[4]
        
        # Combine into a dataframe
        df = pd.DataFrame(columns=["Board State", "Winner", "Improved Priors", "Legal Moves"],
                          data = zip(board_states, winners, improved_priors, legal_moves))
        
        # Check the save folder exists. If not, make it
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        
        save_path = save_folder / f"{self.model}.{self.model.gen}_{num_games}_games_{num_simulations}_MCTS_sims.pkl"

        # Save as pickle to maintain lists as lists (csv converts them to strings)
        df.to_pickle(save_path)

    def generate_parallel_games(self, num_simulations, num_games, save_folder=Path(os.getcwd()) / "generated_games"):
        start_state = (0, 0, 0, 0)
        games = {i : GameInfo(state=start_state) for i in range(num_games)} # setup games
        incomplete_games = {key : val for key, val in games.items()} # duplicate games

        calculated_positions = {}
        while len(incomplete_games) > 0:
            # Extract the states and run
            states = {game_info.state for game_info in incomplete_games.values()}
            roots, calculated_positions = self.mcts.run_parallel(states,
                                                                 to_play=1,
                                                                 num_simulations=num_simulations,
                                                                 calculated_positions=calculated_positions)

            # update the nodes
            for game_info in incomplete_games.values():
                game_info.tree = roots[game_info.state]
                game_info.update_improved_priors()
                game_info.update_legal_moves()
                action = game_info.select_action() # also saves the action choice in game_info.actions
                game_info.get_next_state(action, self.game) # makes the action, reverses board view and updates board_states
                game_info.update_outcome(self.game)
            
            # update incomplete_games
            incomplete_games = {key : val for key, val in games.items() if val.outcome is None}
        
        board_states, winners, improved_priors, legal_moves = [], [], [], []
        # Games are now finished. Compute winner vector and save
        for game_info in games.values():
            #print(game_info.board_states)
            board_states += game_info.board_states[:-1] # includes final state (which we don't want), so use list slice
            winners += game_info.update_winner()[:-1]

            improved_priors += game_info.improved_priors
            legal_moves += game_info.legal_moves
        
        # Combine into a dataframe
        df = pd.DataFrame(columns=["Board State", "Winner", "Improved Priors", "Legal Moves"],
                          data = zip(board_states, winners, improved_priors, legal_moves))
        
        # Check the save folder exists. If not, make it
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        
        save_path = save_folder / f"{self.model}.{self.model.gen}_{num_games}_games_{num_simulations}_MCTS_sims.pkl"

        # Save as pickle to maintain lists as lists (csv converts them to strings)
        df.to_pickle(save_path)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_builder.ConvModelV0(input_shape=4,
                                      hidden_units=16,
                                      output_shape=1,
                                      kernel_size=3).to(device)
    game_generator = GameGenerator(model=model, game_type=connect2.Connect2Game)

    num_games = 1000
    num_sims = 40
    
    serial_start = time.time()
    time_taken = game_generator.generate_n_games(num_games=num_games, num_simulations=num_sims)
    serial_end = time.time()
    print(f"Serially generated and saved {num_games} games in {serial_end - serial_start} seconds")
    
    parallel_start = time.time()
    time_taken = game_generator.generate_parallel_games(num_games=num_games, num_simulations=num_sims)
    parallel_end = time.time()
    print(f"Parallelely generated and saved {num_games} games in {parallel_end - parallel_start} seconds")



    df2 = pd.read_pickle(Path(os.getcwd()) / "generated_games" / f"{model}.1_{num_games}_games_{num_sims}_MCTS_sims.pkl")

    print(df2.head(20))
    