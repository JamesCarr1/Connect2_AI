import torch

from utils import Stack

class Connect2:

    ### (a) - Methods used to get and update game states and outcomes in series
    
    def get_next_state(self, start_state, to_play, action):
        """
        Makes a move. Assumes the move IS possible **** may cause ERRORS
        """
        next_state = [piece for piece in start_state] # must copy into a list to change values
        next_state[action] = to_play
        return tuple(next_state) # then convert back to tuple

    def get_valid_moves_mask(self, game_state):
        """
        Returns a mask of legal moves (i.e empty squares) in a game board.
        """
        return [1 if piece == 0 else 0 for piece in game_state]
    
    def reverse_board_view(self, game_state):
        """
        Takes a game state and reverses the view. Returns a tuple
        """
        return tuple([piece * -1 for piece in game_state])
    
    def get_outcome(self, game_state: list):
        """
        Checks if the game has finished
        """ 
        for i, piece in enumerate(game_state[:-1]):
            if piece == game_state[i+1] and piece != 0: # check if there is a winner
                outcome = piece # Set current player as winner ***** Check player
                return outcome
        
        if game_state.count(0) == 0: # i.e all spaces are taken
            return 0
        
        return None
    
    ### (b) - Methods used to get and update game states in parallel

    def parallel_get_valid_moves_masks(self, game_states: torch.Tensor):
        """
        Returns masks of legal moves in a game positions.
        args:
            game_states: torch.tensor
        """
        res = torch.zeros_like(game_states)

        res[game_states==0] = 1 # If there is no piece, it is a valid move, so make it 1.

        return res


class Connect2Game(Connect2):
    """
    Contains extra methods and attributes used to play a game of Connect2.
    """
    def __init__(self, board_length):
        self.board_length = board_length # number of positions on the board
        self.game_state = [0] * board_length # Setup empty game board

        self.player = 1 # player's turn
        self.outcome = None

        self.available_moves = list(range(self.board_length)) # indices of free squares
        self.moves = Stack() # keeps track of the moves that have been played

    def set_game_state(self, game_state):
        """
        Sets the game's current state.
        """
        if game_state is not None:
            self.game_state = game_state.copy()

    def push(self, move):
        """
        Makes the move for the player.

        args:
            move: Position where the piece is played
        returns:
            None
        """
        if move in self.available_moves:
            self.game_state[move] = self.player # make the move

            if self.get_outcome() is None: # i.e if the game is not finished
                self.update_available_moves()
                self.player *= -1
                self.moves.push(move)
        else:
            raise ValueError
    
    def pop(self):
        """
        Undoes the last move and returns it

        returns:
            last_move: The last move played
        """
        last_move = self.moves.pop()
        self.game_state[last_move] = 0
        return last_move

    def update_available_moves(self):
        """
        Updates the moves that can be played
        """
        if self.game_state is not None:
            self.available_moves = [i for i, piece in enumerate(self.game_state) if piece == 0] # add the move if square is empty (0)
    
    def get_outcome(self):
        """
        Performs Connect2.get_outcome(), but updates self.outcome
        """
        self.outcome = super().get_outcome()

        return self.outcome

if __name__ == '__main__':
    game = Connect2()
    
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
