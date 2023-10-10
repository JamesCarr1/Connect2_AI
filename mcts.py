
import model_builder
import torch
import connect2

import math
import random

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
    model = model_builder.NaiveUnevenModel()
    game = connect2.Connect2Game()

    mcts = MCTS(model, game)

    mcts.run([0, 0, 0, 0], 1, 200)