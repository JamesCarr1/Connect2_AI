import pandas as pd
import torch
import os

from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

class GameDataset(Dataset):
    def __init__(self, game_file):
        # Open the data using pandas
        self.data = pd.read_pickle(game_file)

        # Unpack the relevant data columns
        self.board_states = torch.tensor(self.data["Board State"].to_list(), dtype=torch.float32) # input into the model
        self.winners = torch.tensor(self.data["Winner"].to_list(), dtype=torch.float32) # target output of the value head

        # These are of varying length, so cannot be converted to tensors yet
        self.target_priors = self.data["Improved Priors"].to_list() # target priors (i.e those produced by the prior improvement policy)
        self.legal_moves = self.data["Legal Moves"].to_list() # legal moves, allows reconstruction of target priors to full length
    
    def __len__(self):
        return len(self.board_states)
    
    def __getitem__(self, idx):
        # First need to expand target_priors
        expanded_target_priors = torch.tensor(self.target_priors[idx], dtype=torch.float32)

        # Convert legal moves to a tensor by appending -1s
        legal_moves_tensor = torch.tensor(self.legal_moves[idx], dtype=torch.int64)
        
        expanded_legal_moves = torch.concat([
            legal_moves_tensor, # the data itself
            torch.zeros(4 - len(legal_moves_tensor)), # appends zeros so all tensors are length 4 (+ 1 for the len below)
            torch.tensor([len(legal_moves_tensor)]) # appends len so legal_moves_tensor can be recovered
        ])

        # Now expand with zeros
        expanded_target_priors = torch.zeros(4).scatter(dim=0, index=legal_moves_tensor, src=expanded_target_priors)

        # Now get other data
        board_state = self.board_states[idx]
        winner = self.winners[idx]
        torch.tensor([]).renorm

        # Return input (board_state and legal moves) and the targets for the two outputs - expanded_target_priors and winner
        return board_state, expanded_legal_moves, expanded_target_priors, winner
    
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