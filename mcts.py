
import model_builder
import torch
import connect2

import math
import random
import time

import numpy as np

class Node:
    def __init__(self, prior, to_play, depth=0):
        self.prior = prior # The probability of selecting this state from it's parents
        self.depth = depth # depth down the tree.
        self.to_play = to_play # The player whose turn it is (-1 or 1)
        self.state = None # The TRUE state of the board (irrespective of view)

        self.children = {}

        self.visit_count = 0 # Number of times the state has been visited
        self.value_sum = 0 # The total value of this state from all visits

        self.outcome = None # The outcome at this node (1, 0 or -1 for the three outcomes). None for unfinished game

    def __repr__(self):
        """
        Print format for the node
        """
        return f'{self.state}: prior: {self.prior}, value: {self.value()}, visit_count: {self.visit_count}'

    def value(self):
        # Returns an averaged value
        if self.visit_count > 0:
            return self.value_sum / self.visit_count
        else:
            return 0
    
    def expand(self, state, action_probs):
        """
        Expands the tree with all possible moves from the current state.
        args:
            state: state of the board at this node
            action_probs: probability of each move. Masked to remove impossible moves
        """
        self.state = state # set this node's state

        for i, prob in enumerate(action_probs):
            if prob != 0: # i.e move is possible
                self.children[i] = Node(prob, self.to_play * -1, self.depth + 1) # other player's turn so * -1

    def expanded(self):
        """
        Checks if node has been expanded i.e if children has elements
        """
        return len(self.children) != 0 # if there ARE elements in children, returns true

    def select_action(self, choose_type):'''
        """
        Randomly chooses an action, using the value as weights
        
        args:
            type: 'greedy' or 'random'. 'greedy' selects the action with the highest value. 'ucb' selects the action with the highest ucb score
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
            '''
    
    def select_child(self):
        """
        Selects the child with the highest UCB score
        """
        priors = [node.prior for node in self.children.values()]

        # Setup variables with default values
        best_score = -np.inf
        best_action = -1
        best_child = None

        # Cycle through children and choose the one with the highest ucb score
        for action, child in self.children.items():
            score = self.ucb_score(child)
            if score > best_score:
                # Update variables
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def select_action(self, choose_type):
        """
        Selects the child, either greedily or randomly

        args:
            type: 'greedy' or 'random'. 'greedy' selects the action with the most visits. 'random' samples with visit_counts as weights
        """
        visit_counts = [node.visit_count for node in self.children.values()] # get visit counts
        actions = [action for action in self.children.keys()]

        if choose_type == 'greedy':
            ### NEED TO MAKE MORE EFFICIENT
            visits_actions = zip(visit_counts, actions) # zip up the two lists
            best_action = max(visits_actions, key=lambda x: x[0])[1] # Choose the action (x[1]) with the highest visit count (x[0])
            return best_action
        elif choose_type == 'random':
            action = random.choices(actions, weights=visit_counts)[0] # choose the action randomly
            return action

    def ucb_score(self, child):
        """
        Implementation of ucb score that only takes in one value - the child (parent is self)
        """
        return ucb_score(self, child)

    def plot(self):
        """
        Plots the current tree structure extending from this node
        """
        print(f"{'|---' * self.depth}{self}")
        for child in self.children.values():
            child.plot()

class NodeInfo:
    """
    Holds information about a given search as part of MCTS.run_parralel()

    attr:
        node: Node: the bottom of the tree currently being searched
        search_path: list[Node] - the nodes traversed to reach an unexpanded node in the tree
        action: int - the action taken to move from parent.state to new_state
        parent: Node - the parent node of the current node of interest
        next_state: the position after action has been played from parent.state
        value: the outcome of the match or model prediction
        action_probs: the probability of each action according to the model
    """
    def __init__(self,
                 node=None,
                 action=None,
                 parent=None,
                 next_state=None,
                 value = None,
                 action_probs=None):
        self.node = node
        self.search_path = [self.node]
        self.action = action
        self.parent = parent
        self.next_state = next_state

        self.value = value
        self.action_probs = action_probs
    
    def to_unexpanded(self):
        """
        Performs the node.select_child() method until an unexpanded node is reached, updating internal attributes along the way
        """
        while self.node.expanded():
            # Choose an action
            self.action, self.node = self.node.select_child()
            # Save the new node to search path
            self.search_path.append(self.node)

    def expand(self):
        """
        Expands the current node using next state and action probs
        """
        self.node.expand(self.next_state, self.action_probs)

    def update_parent(self):
        self.parent = self.search_path[-2]
    
    def get_next_state(self, game):
        """
        From the parent state, makes self.action and flips the board view
        args:
            game: the game instance used for this tree search
        """
        self.next_state = game.get_next_state(start_state=self.parent.state, # make the move
                                              to_play=1,
                                              action=self.action)
        self.next_state = game.reverse_board_view(self.next_state) # and reverse board view
    
    def get_outcome_for_player(self, game):
        """
        Checks if the game is over - if so, update value
        """
        outcome = game.get_outcome_for_player(self.next_state, player=1)
        if outcome is not None:
            self.value = outcome
    
    def backpropogate(self):
        for node in reversed(self.search_path):
            node.value_sum += self.value if node.to_play == self.parent.to_play else -self.value
            node.visit_count += 1

    def update_results(self, results):
        """
        Extracts the information given from calculated_positions dictionary.
        args:
            results: tuple(action_probs, value) - note value is in list form, so also needs to be extracted
        """
        action_probs, value = results

        self.action_probs = action_probs
        self.value = value[0] # value is is list

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
        action_probs, _ = self.predict_mask_and_normalise(state) # don't need value
        root.expand(state, action_probs)

        #root.plot()

        # Now simulate num_simulation times
        for _ in range(num_simulations):
            node = root # always start from the root
            search_path = [node] # contains the nodes searched in this simulation

            while node.expanded(): # get to an unexpanded node
                action, node = node.select_child() # action: the move required to get from the original node to the new node
                search_path.append(node)

            parent = search_path[-2] # save parent of node
            state = parent.state

            # Now get next state (from PARENT'S POV)
            next_state = self.game.get_next_state(state, to_play=1, action=action)

            # Flip board view (to CHILD'S POV)
            next_state = self.game.reverse_board_view(next_state)

            # Checks if game is over
            value = self.game.get_outcome_for_player(next_state, player=1)

            # If game is NOT over:
            if value is None:
                # Expand the node
                action_probs, value = self.predict_mask_and_normalise(next_state)
                node.expand(next_state, action_probs)
            
            # Now backpropogate
            self.backpropogate(search_path, value, parent.to_play * -1)
            #root.plot()
        
        # Can uncomment to plot final tree
        #root.plot()

        return root

    def run_parallel(self, states, to_play, num_simulations=5, choose_type='random', calculated_positions={}):
        """
        Runs num_games Monte Carlo tree seaches, starting at states.
        args:
            states: list containing the state at the root of each tree. len = num_games. Note states must be a list of TUPLES
            to_play: the player whose turn it is.
            num_simulations: the depth of the MCTS
            choose_type: 'random' == ucb_score, 'greedy' == exploit term in ucb_score
            calculated_positions: board_states which have already been calculated
        returns:
            roots: dict containing all the root nodes
        """

        # Find all unique starting positions
        ### COULD INVESTIGATE USING torch.unique, dim=0!!
        unique_states = set(states)
        # Find all states which have NOT been passed through the model
        new_states = [state for state in unique_states if state not in calculated_positions.keys()]
        
        # Initialise num_games root nodes. MC trees are deterministic given the starting state, so only need to create
        # one for each unique state
        roots = {state: Node(0, to_play) for state in unique_states}

        # Make prediction
        action_probs, values = self.parallel_predict_mask_and_normalise(new_states)
        # zip them up
        results = zip(action_probs, values)
        # And add them to the dictionary
        calculated_positions |= dict(zip(new_states, results))

        # Now expand all root nodes
        for i, state in enumerate(unique_states):
            action_probs, _ = calculated_positions[state]
            roots[state].expand(state, action_probs)
        
        roots[0, 0, 0, 0].plot()

        # Now simulate num_simulations times
        for _ in range(num_simulations):
            # Setup the bases of the trees
            nodes = roots
            start_states = nodes.keys()
            search_paths = {state: [node] for state, node in nodes.items()}
            actions = {}

            # Now get to unexpanded nodes
            for state in nodes:
                while nodes[state].node.expanded():
                    # Choose an action
                    action, nodes[state].node = nodes[state].select_child()
                    # Save the node to search path
                    search_paths[state].append(nodes[state])
                # Save the final action chosen
                actions[state] = action
            
            # And save all parents
            parent_to_play = list(search_paths.values())[0][-2].to_play
            parents = {start_state: search_path[-2].state for start_state, search_path in search_paths.items()}

            # And get all the next states
            next_states = self.game.parallel_get_next_states(parents.values(), to_play=1, actions=actions) # note the output is a tensor
            next_states = self.game.parallel_get_next_states(parents, to_play, actions)
            ############# NEW CLASS TO HOLD ALL THIS INFO (actions, parent, parent state, value etc.)
            # and flip board view
            next_states = self.game.parallel_reverse_board_view(next_states) # note the output is a list

            # Create a translation reference from parent state to next state
            children = dict(zip(parents.keys(), next_states))

            # Now find all new states
            new_states = {state for state in next_states if state not in calculated_positions}

            # Find if games have finished
            values = {}
            new_not_finished = [] # list of states which are not finished
            for new_state in enumerate(new_states):
                values[new_state] = self.game.get_outcome_for_player(new_state, to_play=1) # get the results
                if values[new_state] is None:
                    new_not_finished.append(state) # create a list of games which have not finished, and are new states
                else:
                    calculated_positions[new_state] = values[new_state]
            
            # Calculate these positions
            results = zip(self.parallel_predict_mask_and_normalise(new_not_finished))

            # and add to the dictionary
            calculated_positions |= dict(zip(new_not_finished, results))

            # Now go back through and update values dict
            for new_state in new_states:
                if values[new_state] is None:
                    # update with calculated value
                    values[new_state] = calculated_positions[new_state]
            
            # Now go through and create a new dict with the results
            node_values = {}
            for start_state in nodes: # cycle through the starting states
                # find the parent state
                parent_state = parents[start_state]
                # find the new state
                new_state = children[start_state]
                # and add the result
                node_values[start_state] = calculated_positions[new_state]

            # Now backpropogate all nodes
            for start_state in nodes:
                self.backpropogate(search_path=search_paths[start_state],
                                   value=node_values[start_state],
                                   to_play=parent_to_play * -1)
            roots[(0, 0, 0, 0)].plot()
        
        roots[(0, 0, 0, 0)].plot()
        return roots

    def run_parallel(self, states, to_play, num_simulations=5, choose_type='random', calculated_positions={}):
        """
        Runs num_games Monte Carlo tree seaches, starting at states.
        args:
            states: list containing the state at the root of each tree. len = num_games. Note states must be a list of TUPLES
            to_play: the player whose turn it is.
            num_simulations: the depth of the MCTS
            choose_type: 'random' == ucb_score, 'greedy' == exploit term in ucb_score
            calculated_positions: board_states which have already been calculated
        returns:
            roots: dict containing all the root nodes
        """

        # Find all unique starting positions
        ### COULD INVESTIGATE USING torch.unique, dim=0!!
        unique_states = set(states)
        # Find all states which have NOT been passed through the model
        new_states = [state for state in unique_states if state not in calculated_positions.keys()]
        
        # Initialise num_games root nodes. MC trees are deterministic given the starting state, so only need to create
        # one for each unique state
        roots = {state: Node(0, to_play) for state in unique_states}
        if len(new_states) > 0:
            # Make prediction
            action_probs, values = self.parallel_predict_mask_and_normalise(new_states)
            results = zip(action_probs, values)
            # And add them to the dictionary
            calculated_positions |= dict(zip(new_states, results))

        # Now expand all root nodes
        for state in unique_states:
            action_probs, _ = calculated_positions[state]
            roots[state].expand(state, action_probs)
        
        #roots[0, 0, 0, 0].plot()

        # Now simulate num_simulations times
        for _ in range(num_simulations):
            # Setup the info object to contain all info about each tree search
            nodes = {state: NodeInfo(node=roots[state]) for state in roots} # now have a translation from state to all state info

            # Perform tree search setup on each NodeInfo
            for node_info in nodes.values():
                node_info.to_unexpanded()
                node_info.update_parent()
                node_info.get_next_state(self.game) # this includes making the move AND flipping the board
                node_info.get_outcome_for_player(self.game) # check if the game is over
            
            # Search through node_infos and add the ones with uncalculated positions to a set. Only choose positions which have 
            # have no outcome yet
            new_states = {ni.next_state for ni in nodes.values() if (ni.next_state not in calculated_positions) and (ni.value is None)}
            new_states = list(new_states) # torch.tensor does not understand sets
            
            # Calculate these positions and add to the dictionary
            if len(new_states) > 0:
                action_probs, values = self.parallel_predict_mask_and_normalise(new_states)
                results = zip(action_probs, values)
                calculated_positions |= dict(zip(new_states, results))

            # Go back through and update values and action probs. Then expand the node
            for node_info in nodes.values():
                if node_info.value is None:
                    node_info.update_results(calculated_positions[node_info.next_state])
                    node_info.expand()
            
            # Backpropogate all nodes
            for node_info in nodes.values():
                node_info.backpropogate()
            
            #roots[(0, 0, 0, 0)].plot()
        
        #roots[(0, 0, 0, 0)].plot()

        return roots, calculated_positions

    def predict_mask_and_normalise(self, state):
        """
        Uses model to predict priors, masks out the illegal moves and renormalises
        """
        # predict
        action_probs, value = self.model.predict(state) # predict priors

        # mask
        mask = self.game.get_valid_moves(state) # mask illegal moves
        action_probs = [prob * mask_val for prob, mask_val in zip(action_probs, mask)] # calculate probs
        
        # and normalise
        total = sum(action_probs)
        action_probs = [prob / total for prob in action_probs]

        return action_probs, value
    
    def parallel_predict_mask_and_normalise(self, states):
        """
        Uses model to predict priors, masks out illegal moves and renormalises
        """

        # Convert to a tensor
        states = torch.tensor(states, dtype=torch.float32)

        # predict. Note this returns a TENSOR (as bitwise multiplication is easier)
        action_probs, values = self.model.parallel_predict(states)

        # mask
        masks = self.game.parallel_get_valid_moves(states)
        action_probs = action_probs * masks

        # and normalise
        # Sum along the 1st axis. Unsqueeze ensures the sum divides every row, dim=0 would 
        # divide every column by the corresponding element of the sum
        sums = torch.sum(action_probs, dim=1).unsqueeze(dim=1)
        action_probs = torch.div(action_probs, sums)

        return action_probs.tolist(), values.tolist()



    def backpropogate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

def ucb_score(parent: Node, child: Node, c_ucb=4):
    """Calculates 'ucb score', which can be used as weights for the MCTS sampling.
    See: 
    https://www.chessprogramming.org/UCT 
    https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
    for more info
    
    args:
        c_ucb: used to weight/deweight the exploration term. Default 4.
    """
    # This term encourages exploration. Nodes with few visits (child.visit_count is low) will have higher values than nodes with many visits.
    explore_score = c_ucb * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    
    # This term encourages exploitation. Nodes with a high win ratio (in this case predicted win ratio) will have high values
    if child.visit_count > 0:
        # The value of the child is from the opposing player's POV
        exploit_score = -child.value()
    else:
        exploit_score = 0
    
    return explore_score + exploit_score

if __name__ == '__main__':
    model = model_builder.ConvModelV0(input_shape=4, hidden_units=16, output_shape=1, kernel_size=3).to('cpu')
    game = connect2.Connect2Game()

    mcts = MCTS(model, game)

    # Parallel run 1000 'games' starting at (0, 0, 0, 0):
    #states = [(0, 0, 0, 0)] * 1000
    states = [(-1, 0, 0, 0),
                (0, -1, 0, 0),
                           (0, 0, -1, 0),
                           (0, 0, 0, -1)] * 250
    parallel_start = time.time()
    mcts.run_parallel([(0, 0, 0, 0)], 1, num_simulations=40)
    parallel_end = time.time()

    # Serial run 1000 'games' starting at (0, 0, 0, 0)
    serial_start = time.time()
    for state in states:
        mcts.run(state, to_play=1, num_simulations=40)
    serial_end = time.time()

    print(f"Parallel method took {parallel_end - parallel_start} seconds")
    print(f"Serial method took {serial_end - serial_start} seconds")