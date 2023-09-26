import os
import torch

import prepare_data
import model_builder
import engine
import utils

from pathlib import Path
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Setup device-agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup dataloaders
    file_path = Path(os.getcwd()) / "generated_games" / "LinearModelV0.1_200_games_7_MCTS_sims.pkl"
    train_dataset, test_dataset, train_dataloader, test_dataloader = prepare_data.prepare_dataloaders(file_path=file_path,
                                                                                                      num_workers=1)

    # Setup model
    input_shape = 4
    hidden_units = 2
    output_shape = 1
    model = model_builder.LinearModelV0(input_shape=input_shape,
                                        hidden_units=hidden_units,
                                        output_shape=output_shape)

    # Setup loss functions and optimizer
    value_loss = torch.nn.MSELoss()
    prior_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.01)
    
    # And train
    epochs = 5
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