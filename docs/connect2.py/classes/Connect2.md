Class containing the functionality to play a game of connect 2.
## Methods

### The methods are split into two groups:
#### (a) - Methods used to get and update game states and outcomes in series (one after another)

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