import os
import torch

import prepare_data
import model_builder
import engine
import utils
import data_generator
import connect2

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
                       optimizer,
                       device,
                       scheduler=None):
    """
    Generates a dataset, then trains from it. Saves the updated model and then repeats for num_gens.
    """
    # Setup game generator
    game_generator = data_generator.GameGenerator(model=model, game_type=connect2.Connect2Game)
    for i in range(num_gens):
        # Generate game data
        game_generator.generate_n_games(num_games=num_games, num_simulations=num_sims)
        print(f"Generated games for Generation {i+1}")

        # Setup dataloaders
        file_path = Path(os.getcwd()) / "generated_games" / f"{model}.{i+1}_{num_games}_games_{num_sims}_MCTS_sims.pkl"
        train_dataset, test_dataset, train_dataloader, test_dataloader = prepare_data.prepare_dataloaders(file_path=file_path,
                                                                                                      num_workers=1)
        
        # And train
        results = engine.train(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    value_loss_fn=value_loss_fn,
                    prior_loss_fn=prior_loss_fn,
                    epochs=epochs,
                    device=device,
                    accuracy_fn=utils.value_acc)
        
        # Update model gen and save
        model.gen += 1
        save_path = Path(os.getcwd()) / "models" / f"{model}.{model.gen}.pt"
        torch.save(model.state_dict(), save_path)

        if scheduler is not None:
            scheduler.step()



if __name__ == '__main__':
    # Setup device-agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup model
    input_shape = 4
    hidden_units = 2
    output_shape = 1
    model = model_builder.LinearModelV0(input_shape=input_shape,
                                        hidden_units=hidden_units,
                                        output_shape=output_shape).to(device)

    # Setup loss functions and optimizer
    value_loss_fn = torch.nn.MSELoss()
    prior_loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.05)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    generate_and_train(model=model,
                       num_games=1000,
                       num_sims=10,
                       num_gens=10,
                       epochs=4,
                       value_loss_fn=value_loss_fn,
                       prior_loss_fn=prior_loss_fn,
                       optimizer=optimizer,
                       device=device)
    
    """
    # Setup dataloaders
    file_path = Path(os.getcwd()) / "generated_games" / "LinearModelV0.1_1000_games_15_MCTS_sims.pkl"
    train_dataset, test_dataset, train_dataloader, test_dataloader = prepare_data.prepare_dataloaders(file_path=file_path,
                                                                                                      num_workers=1)

    # And train
    epochs = 100
    results = engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 optimizer=optimizer,
                 value_loss_fn=value_loss,
                 prior_loss_fn=prior_loss,
                 epochs=epochs,
                 device=device,
                 accuracy_fn=utils.value_acc)

    results_to_plot = [("value_train_loss", 'r', '-'),
                       ("prior_train_loss", 'b', '-'),
                       ("total_train_loss", 'g', '-'),
                       ("value_test_loss", 'r', '--'),
                       ("prior_test_loss", 'b', '--'),
                       ("total_test_loss", 'g', '--')]
    
    utils.plot_loss_curves(results=results, labels=results_to_plot)
    plt.show()
    """