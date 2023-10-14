import pandas as pd
import torch
import os

from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

class GameDataset(Dataset):
    def __init__(self, game_file):
        # Open the data using pandas
        self.data = pd.read_pickle(game_file)   

        # Unpack the relevant data columns. NOTE: target_priors (produced by the prior improvement policy) have no relation to the
        # original priors and so are already masked (MCTS can't 'visit' positions where a piece already exists) -> these positions
        # have zero visits -> zero improved prior.
        self.board_states = torch.tensor(self.data["Board State"].to_list(), dtype=torch.float32) # input into the model
        self.winners = torch.tensor(self.data["Winner"].to_list(), dtype=torch.float32) # target output of the value head
        self.target_priors = torch.tensor(self.data["Improved Priors"].to_list(), dtype=torch.float32) # target priors

        # These are of varying length, so cannot be converted to tensors
        self.legal_moves = self.data["Legal Moves"].to_list() # legal moves, allows reconstruction of target priors to full length

        
        self.legal_moves_masks = torch.zeros((len(self.legal_moves), 4)) # used to mask illegal move priors
        for i, legal_move in enumerate(self.legal_moves): # cycle each row and add 1 to legal move positions
            self.legal_moves_masks[i][legal_move] = 1

    
    def __len__(self):
        return len(self.board_states)
    
    def __getitem__(self, idx):
        return self.board_states[idx], self.legal_moves_masks[idx], self.target_priors[idx], self.winners[idx]
    
def prepare_dataloaders(file_path,
                        num_workers=os.cpu_count(),
                        batch_size=32,
                        transform=None):
    """
    Obtains data from files and forms train and test dataloaders

    args:
        file_path: Path object - path where data is stored
        num_workers
        batch_size
    returns:
        train_dataset
        test_dataset
        train_dataloader
        test_dataloader
    """
    # Open up all the data together
    dataset = GameDataset(file_path)

    # Split into train (80%) and test (20%) datasets
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    # And setup dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)
    
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader

if __name__ == '__main__':
    file_path = Path(os.getcwd()) / "generated_games" / "LinearModelV0.1_200_games_7_MCTS_sims.pkl"

    train_dataset, test_dataset, train_dataloader, test_dataloader = prepare_dataloaders(file_path=file_path)

    for board_state, expanded_legal_moves, expanded_target_priors, winner in train_dataloader:
        print(expanded_legal_moves.shape)

        for row in expanded_legal_moves:
            print(row.shape)