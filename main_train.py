import os
import numpy as np
import random
import data_processing
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch
import copy
from model import LearnedSimulator
import functional
from tqdm import tqdm
import argparse

PARAMS_REAL = {
    'run_name': 'real_entire_dataset',
    'dataset_dir': ['data/10Nodes_by_tree/trial', 'data/20Nodes_by_tree/trial'],
    'num_trees_per_dir': [27,43],
    'simulated_dataset': False,
    'num_epochs': 700,
    'batch_size': 512,
    'lr': 2e-3,
    'train_validation_split': 0.9,
    'remove_duplicates': True
}

PARAMS_SIM = {
    'run_name': 'sim_entire_dataset',
    'dataset_dir': ['data/tree_dataset/trial'],
    'num_trees_per_dir': [10],
    'simulated_dataset': True,
    'num_epochs': 700,
    'batch_size': 512,
    'lr': 2e-3,
    'train_validation_split': 0.9,
    'remove_duplicates': True
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sim')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Bells and whistles.
    params = PARAMS_SIM if args.data == 'sim' else PARAMS_REAL
    params['seed'] = args.seed
    output_dir = 'output/{}'.format(params['run_name'])
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    # X and Y mean vertex positions before and after force is applied.
    X_force_list = []
    X_pos_list = []
    Y_pos_list = []
    train_dataset = []
    val_dataset = []

    # NOTE(daniel): real data starts from trial1 instead of trial0.
    offset = 0 if args.data == 'sim' else 1

    for i_dir, dataset_dir in enumerate(params['dataset_dir']):
        train_val_split = int(params['num_trees_per_dir'][i_dir]*params['train_validation_split'])

        for i in tqdm(range(offset, params['num_trees_per_dir'][i_dir] + offset)):
            d = dataset_dir+str(i)
            X_edges, X_force, X_pos, Y_pos = data_processing.load_npy(d, params['simulated_dataset'])
            if params['remove_duplicates']:
                X_edges, X_force, X_pos, Y_pos = data_processing.remove_duplicate_nodes(
                        X_edges, X_pos, Y_pos, X_force)

            if i < train_val_split:
                train_dataset += data_processing.make_dataset(X_edges, X_force, X_pos, Y_pos, i_dir,
                                make_directed=True, prune_augmented=False, rotate_augmented=False)
            else:
                val_dataset += data_processing.make_dataset(X_edges, X_force, X_pos, Y_pos, i_dir,
                                make_directed=True, prune_augmented=False, rotate_augmented=False)

    print('Train dataset size: {}'.format(len(train_dataset)))
    print('Validation dataset size: {}'.format(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LearnedSimulator().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, min_lr=5e-4)

    train_loss_history = []
    val_loss_history = []
    best_loss = 1e9
    try:
        for epoch in range(1, params['num_epochs']+1):
            train_loss = functional.train(model, optimizer, criterion, train_loader, epoch, device)
            val_loss = functional.validate(model, criterion, val_loader, epoch, device)
            if val_loss<best_loss:
                best_loss=val_loss
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            scheduler.step(best_loss)
            print('Epoch {} | Train Loss: {} | Val Loss: {} | LR: {}'.format(epoch, train_loss, val_loss, scheduler._last_lr))
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            with open(os.path.join(output_dir, 'trajectory.txt'), 'a') as file1:
                file1.write('{} {} {} {}\n'.format(epoch, train_loss, val_loss, scheduler._last_lr))

        ax.plot(train_loss_history, 'r', label='train')
        ax.plot(val_loss_history, 'b', label='validation')
        ax.legend(loc="upper right")
        plt.show()
        torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
    except KeyboardInterrupt:
        ax.plot(train_loss_history, 'r', label='train')
        ax.plot(val_loss_history, 'b', label='validation')
        ax.legend(loc="upper right")
        plt.show()
        torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
