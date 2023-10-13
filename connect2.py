import torch

class Connect2Game:
    def __init__(self, board_length=4):
        self.board_length = board_length # number of positions on the board
        self.game_state = [0] * board_length # Setup empty game board

        self.outcome = None # 1 for p1 win, 0 for draw, -1 for p2 win
        self.player = 1 # player's turn

        self.available_moves = [0, 1, 2, 3] # indices of free squares
        self.moves = [] # keeps track of the moves that have been played
    
    def push(self, move):
        """
        Makes the move for the player.

        args:
            move: index of the board where the piece is played
            player: player who plays the move (0 or 1)
        returns:
            None
        """
        if move in self.available_moves:
            self.game_state[move] = self.player # make the move
            if not self.get_outcome(): # i.e if the game is not over, update player and available moves
                self.update_available_moves()
                self.player *= -1 # update the current player
                self.moves.append(move)
        else:
            raise ValueError
    
    def pop(self):
        # Undoes the last move
        last_move = self.moves.pop(-1)
        self.game_state[last_move] = 0
        return last_move

    def set_game_state(self, game_state):
        if game_state is not None:
            self.game_state = game_state.copy()
        self.update_available_moves()
    '''
    def get_next_state(self, game_state, to_play, action):
        """
        Sets the board in the current state (game_state) and plays the action

        args:
            game_state: The current game_state at the moment of interest
            to_play: the player playing the move
            action: the index of the board where the piece is played
        returns:
            next_state: the game_state after action
        """
        self.set_game_state(game_state)
        if action in self.available_moves:
            self.game_state[action] = to_play
            return self.game_state
        else:
            raise ValueError
    '''
    def get_next_state(self, start_state, to_play, action):
        """
        Makes a move. Assumes the move IS possible
        """
        self.set_game_state(start_state)
        self.game_state[action] = to_play
        return self.game_state

    def get_next_state(self, start_state, to_play, action):
        """
        Makes a move. Assumes the move IS possible **** may cause ERRORS
        """
        next_state = [piece for piece in start_state] # need to make a list to change
        next_state[action] = to_play
        return tuple(next_state) # then convert back to tuple
    
    def parallel_get_next_states(self, game_states, to_play, actions):
        """
        Makes the moves in actions. Assumes the moves ARE possible
        """
        # Convert to tensor. Easier to use for scatter
        game_states = torch.tensor(game_states)
        # Unsqueeze converts it to the indices on each row that must be changed.
        # e.g: [1, 2, 0] -> [[1], [2], [0]] i.e change the 1th element of row 0, the 2th of row 1 and 0th of row 2
        index = torch.tensor(actions).unsqueeze(dim=1)

        # scatter the moves in
        game_states.scatter_(dim=1, index=index, src=to_play * torch.ones_like(game_states))

        # return as a list
        return game_states

    def reverse_board_view(self, game_state):
        """
        Takes a game state and reverses the view

        args:
            game_state: current state (from p1's view)
        returns:
            reversed_view: current state (from p2's view)
        """
        return [piece * -1 for piece in game_state] 

    def reverse_board_view(self, game_state):
        """
        Takes a game state and reverses the view. Returns a tuple

        args:
            game_state: current state (from p1's view)
        returns:
            reversed_view: current state (from p2's view)
        """
        return tuple([piece * -1 for piece in game_state])
    
    def parallel_reverse_board_view(self, game_states: torch.tensor):
        """
        Takes many games and reverses the view. Note game_states is a torch.tensor
        args:
            game_states: torch.tensor
        returns:
            reversed_game_states: list
        """

        return (game_states * -1).tolist()

    def update_available_moves(self):
        """
        Updates the moves that can be played
        """
        if self.game_state is not None:
            self.available_moves = [i for i in range(self.board_length) if self.game_state[i] == 0] # Add the move if self.game_state[i] == 0
    
    def get_outcome_for_player(self, game_state: list, player):
        """
        Checks if the game has finished
        """ 
        for i in range(self.board_length - 1):
            if game_state[i] == game_state[i+1] and game_state[i] != 0: # check if there is a winner
                self.outcome = self.player # Set current player as winner ***** Check player
                return self.outcome
        
        if game_state.count(0) == 0: # i.e all spaces are taken
            return 0
        
        return None
    
    def get_valid_moves(self, game_state):
        """
        Returns a mask of legal moves in a game board.
        """
        return [1 if game_state[i] == 0 else 0 for i in range(4)]

    def parallel_get_valid_moves(self, game_states: torch.Tensor):
        """
        Returns masks of legal moves in a game positions.
        args:
            game_states: torch.tensor
        """
        res = torch.zeros_like(game_states)

        res[game_states==0] = 1 # If there is no piece, it is a valid move, so make it 1.

        return res

if __name__ == '__main__':
    game = Connect2Game()
    
    while game.outcome is None:
        print(f"Game board: {game.game_state}")
        while True:
            try:
                move = int(input(f"Player {game.player}: Enter move: "))
                game.push(move)
                break
            except ValueError:
                print("Invalid move - retry")
    
    print(f"Game over: Player {game.outcome} won.")
