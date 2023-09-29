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
                       value_optimizer,
                       prior_optimizer,
                       device,
                       alpha=0.5,
                       schedulers=None):
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
                    accuracy_fn=utils.value_acc,
                    alpha=alpha)
        
        
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

    value_optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.01)
    prior_optimizer = torch.optim.Adam(params=model.parameters(),
                                       lr=0.1)
    
    value_scheduler = lr_scheduler.ExponentialLR(value_optimizer, gamma=0.99)
    prior_scheduler = lr_scheduler.ExponentialLR(prior_optimizer, gamma=0.99)

    
    # results = generate_and_train(model=model,
    #                    num_games=1000,
    #                    num_sims=30,
    #                    num_gens=100,
    #                    epochs=1,
    #                    value_loss_fn=value_loss_fn,
    #                    prior_loss_fn=prior_loss_fn,
    #                    value_optimizer=value_optimizer,
    #                    prior_optimizer=prior_optimizer,
    #                    device=device,
    #                    schedulers=(value_scheduler, prior_scheduler))

    results = generate_and_train(model=model,
                       num_games=1000,
                       num_sims=10,
                       num_gens=10,
                       epochs=1,
                       value_loss_fn=value_loss_fn,
                       prior_loss_fn=prior_loss_fn,
                       value_optimizer=value_optimizer,
                       prior_optimizer=prior_optimizer,
                       device=device,
                       schedulers=[value_scheduler],
                       alpha=0.5)
    
    test_vector = torch.tensor([[1, 0, 0, -1],
                                [-1, 0, 0, 0],
                                [0, 0, 1, -1],
                                [-1, 0, 0, 1],
                                [0, -1, 0, 0]], dtype=torch.float32).to(device)
    target_priors = torch.tensor([[0, 0.72979, 0.27020, 0],
                                  [0, 0.41657, 0.25619, 0.327],
                                  [0.26969, 0.73030, 0, 0],
                                  [0, 0.28705, 0.71294, 0],
                                  [0.27320, 0, 0.33004, 0.396]])
    target_values = torch.tensor([0, -1, 1, 1, -1])
    
    action_logits, value_logits = model(test_vector)

    pred_priors = torch.softmax(action_logits, dim=1)

    log_softmaxed = torch.log_softmax(action_logits, dim=1)
    pred_values = torch.tanh(value_logits)

    for i, board_state in enumerate(test_vector):
        #print(f"{board_state} | Priors: {qt(target_priors, i)} vs. {qt(pred_priors, i)} | Values: {round(target_values[i].item(), 4)} vs. {round(pred_values[i].item(), 4)}")
        print(f"{board_state} | Priors: {qt(target_priors, i)} vs. {qt(pred_priors, i)} / {qt(action_logits, i)} / {qt(log_softmaxed, i)} | Values: {round(target_values[i].item(), 4)} vs. {round(pred_values[i].item(), 4)}")

    
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
    """
    results_to_plot = [("value_train_loss", 'r', '-'),
                       ("prior_train_loss", 'b', '-'),
                       ("total_train_loss", 'g', '-'),
                       ("value_test_loss", 'r', '--'),
                       ("prior_test_loss", 'b', '--'),
                       ("total_test_loss", 'g', '--')]
    
    utils.plot_loss_curves(results=results, labels=results_to_plot)
    plt.show()
    