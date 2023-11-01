Class containing the functionality to play a game of connect 2.

## Attributes

##### `__init__(board_length=4)`

Initialises an empty game board and all attributes.

- `board_length` - the number of squares on the game board.
- `game_state` - the current state of the game board. Initialised to `[0] * board_length`
- `outcome` - the winner of the game in the current board state.`None` if the game has not finished. `1` for p1 win, `0` for draw, `-1` for p2 win. Initialised to `None`.
- `player` - current player's turn. `1` for p1, `-1` for p2. Initialised to `1`.
- `available_moves` - the indices of free squares (`0`s) on the board. Pieces can be placed on these squares. Initialised to `list(range(board_length))`
- `moves` - the moves that have been made in the current game. Initialised as empty list.

## Methods

### The following methods are split into three groups:
#### (a) - Methods used to get and update game states and outcomes in series (one after another)
##### `set_game_state(game_state)`

Sets the `self.game_state` to `game_state`.

args:
- `game_state`: `list[int]` -  the position to set self.game_state to.

##### `get_next_state(start_state, to_play, action)`

Returns the board position after `action` has been played by `to_play` on `start_state`.

args:
- `start_state`: `tuple[int]` - The game state before the move has been played.
- `to_play`: `int` - The player making the move.
- `action`: `int` - The move made. in this case, the position where the piece is played.
returns:
- `next_state`: `tuple[int]` - The resulting board state.

##### `get_valid_moves_mask(game_state)`

Returns a mask of legal moves (i.e empty squares) in `game_state`.

args:
- `game_state`: `list[int]` or `tuple[int]`
returns:
- `valid_moves_mask`: `list[int]`: mask with `0`s where moves cannot be played and `1s` where they can

##### `reverse_board_view(game_state)`

Takes a game state and reverses the view. Returns a tuple

args:
- `game_state`: `list[int]` or `tuple[int]` - current state (from p1's view)
returns:
- reversed_view: `tuple[int]` - current state (from p2's view)

##### `get_outcome_for_player(game_state)`

Checks if the game has finished and returns the result.

args:
- `game_state`: `list[int]` or `tuple[int]`
returns:
- `outcome`: `int` or `None`

### (b) - Methods used to get and update game states in parallel

##### `parallel_get_valid_moves()`

Returns masks of legal moves in a game positions.

args:
- `game_states`: `torch.tensor` - Current game states
returns:
- `valid_moves_masks`: `torch.tensor` - Masks, where `0`s represent squares where a piece exists (so a move is not valid) and `1`s represent empty squares.
### (c) - Methods used to play the game

##### `push()`

Makes the move for the player.

args:
- `move`: Position where the piece is played.
returns:
- `None`

##### `pop()`

Undoes the last move and returns it

returns:
- `last_move` - The last move played

##### `update_available_moves()`

Updates the moves that can be played.

E.g if the board state changes from [0, 1, 0, 0] -> [0, 1, -1, 0], `self.available_moves` will change from [0, 2, 3] -> [0, 3].