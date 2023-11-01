Superclass of [[Connect2]] that contains extra methods and attributes to play a game of Connect2.

## Attributes
##### `__init__(board_length=4)`

Initialises an empty game board and all attributes.

- `board_length` - the number of squares on the game board.
- `game_state` - the current state of the game board. Initialised to `[0] * board_length`
- `outcome` - the winner of the game in the current board state.`None` if the game has not finished. `1` for p1 win, `0` for draw, `-1` for p2 win. Initialised to `None`.
- `player` - current player's turn. `1` for p1, `-1` for p2. Initialised to `1`.
- `available_moves` - the indices of free squares (`0`s) on the board. Pieces can be placed on these squares. Initialised to `list(range(board_length))`
- `moves` - the moves that have been made in the current game. Initialised as an empty stack (`utils.Stack`).

## Methods

##### `set_game_state(game_state)`

Sets the `self.game_state` to `game_state`.

args:
- `game_state`: `list[int]` -  the position to set self.game_state to.


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

##### `get_outcome()`

Performs `Connect2.get_outcome()`, but also updates self.outcome.