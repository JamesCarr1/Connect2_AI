import os
import torch
import torchmetrics

import prepare_data
import model_builder
import engine
import utils
import data_generator
import connect2
import mcts

import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler

def generate_and_train(model,
                       num_games,
                       num_sims,
                       num_gens,
                       epochs,
                       value_loss_fn,
                       prior_loss_fn,
                       value_optimizer,
                       prior_optimizer,
                       device,
                       value_acc_fn,
                       alpha=0.5,
                       schedulers=None):
    """
    Generates a dataset, then trains from it. Saves the updated model and then repeats for num_gens.
    """
    # Setup game generator
    game_generator = data_generator.GameGenerator(model=model, game_type=connect2.Connect2Game)
    results = None
    for i in range(num_gens):
        # Generate game data
        game_generator.generate_parallel_games(num_games=num_games, num_simulations=num_sims)
        print(f"Generated games for Generation {i+1}")

        # Setup dataloaders
        file_path = Path(os.getcwd()) / "generated_games" / f"{model}.{i+1}_{num_games}_games_{num_sims}_MCTS_sims.pkl"
        train_dataset, test_dataset, train_dataloader, test_dataloader = prepare_data.prepare_dataloaders(file_path=file_path,
                                                                                                      num_workers=1,
                                                                                                      batch_size=128)
        
        # # And train
        # results = engine.train(model=model,
        #             train_dataloader=train_dataloader,
        #             test_dataloader=test_dataloader,
        #             value_optimizer=value_optimizer,
        #             prior_optimizer=prior_optimizer,
        #             value_loss_fn=value_loss_fn,
        #             prior_loss_fn=prior_loss_fn,
        #             epochs=epochs,
        #             device=device,
        #             accuracy_fn=utils.value_acc)

        # And train
        results = engine.train(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=value_optimizer,
                    value_loss_fn=value_loss_fn,
                    prior_loss_fn=prior_loss_fn,
                    epochs=epochs,
                    device=device,
                    value_acc_fn=value_acc_fn,
                    generation=i,
                    alpha=alpha,
                    results=results)
        
        
        # Update model gen and save
        model.gen += 1
        save_path = Path(os.getcwd()) / "models" / f"{model}.{model.gen}.pt"
        torch.save(model.state_dict(), save_path)

        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

    return results

def qt(x, i):
    a = x[i].tolist()

    return [round(y, 4) for y in a]

if __name__ == '__main__':
    # Setup device-agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup model
    input_shape = 4
    hidden_units = 16
    output_shape = 1
    model = model_builder.ConvModelV0(input_shape=input_shape,
                                        hidden_units=hidden_units,
                                        output_shape=output_shape,
                                        kernel_size=3).to(device)

    # Setup loss functions and optimizer
    value_loss_fn = torch.nn.MSELoss()
    prior_loss_fn = torch.nn.CrossEntropyLoss()
    
    # lr 0.01 works well after a bit - maybe should start with lr = 0.005?
    value_optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.01, weight_decay=1e-5) # weight decay provides l2 regularisation
    prior_optimizer = torch.optim.Adam(params=model.parameters(),
                                       lr=0.1)
    
    value_scheduler = lr_scheduler.ExponentialLR(value_optimizer, gamma=0.99)
    prior_scheduler = lr_scheduler.ExponentialLR(prior_optimizer, gamma=0.99)

    value_acc_fn = torchmetrics.Accuracy(task='multiclass', num_classes=3).to(device)
    
    #value_optimizer = torch.optim.Adam(params=model.parameters(),
    #                             lr=0.2, weight_decay=1e-5)
    #value_scheduler = lr_scheduler.MultiStepLR(value_optimizer, gamma=0.1, milestones=[30, 80])
    
    num_games = 1000
    num_sims = 40
    num_gens = 3
    epochs = 1

    results = generate_and_train(model=model,
                       num_games=num_games,
                       num_sims=num_sims,
                       num_gens=num_gens,
                       epochs=epochs,
                       value_loss_fn=value_loss_fn,
                       prior_loss_fn=prior_loss_fn,
                       value_optimizer=value_optimizer,
                       prior_optimizer=prior_optimizer,
                       device=device,
                       value_acc_fn=value_acc_fn,
                       schedulers=[value_scheduler],
                       alpha=0.5)
    
    # Now going to generate some final test data, and print it
    game_gen = data_generator.GameGenerator(model=model, game_type=connect2.Connect2Game)
    #game_gen.generate_n_games(num_simulations=num_sims, num_games=100, save_folder = Path(os.getcwd()) / "test_games")
    game_gen.generate_parallel_games(num_simulations=num_sims, num_games=100, save_folder = Path(os.getcwd()) / "test_games")

    # Now open games
    file_path = Path(os.getcwd()) / "test_games" / f"{model}.{model.gen}_{100}_games_{num_sims}_MCTS_sims.pkl"
    batch_size = 10 # get 10 positions

    test_data = pd.read_pickle(file_path)
    print(test_data.sample(n=batch_size))
    
    train_dataset, test_dataset, train_dataloader, test_dataloader = prepare_data.prepare_dataloaders(file_path=file_path,
                                                                                                      num_workers=1,
                                                                                                      batch_size=batch_size)
    
    # Now cycle through first part of dataloader:
    board_state, legal_moves_mask, target_priors, winner = next(iter(train_dataloader))

    # Get model predictions
    action_logits, value_logits = model(board_state.to(device))
    action_preds = torch.softmax(action_logits, dim=1)
    value_preds = torch.tanh(value_logits)

    # And print them
    for i, board in enumerate(board_state):
        print(f"{board.tolist()} | "
              f"Priors: {qt(target_priors, i)} vs. {qt(action_preds, i)} | "
              f"Values: {round(winner[i].item(), 4)} vs. {round(value_preds[i].item(), 4)}")
    
    results_to_plot = [("value_train_loss", 'r', '-'),
                       ("prior_train_loss", 'b', '-'),
                       ("total_train_loss", 'g', '-'),
                       ("value_test_loss", 'r', '--'),
                       ("prior_test_loss", 'b', '--'),
                       ("total_test_loss", 'g', '--')]
    
    utils.plot_loss_curves(results=results, labels=results_to_plot)
    plt.show()
    
    