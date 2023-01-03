"""Meant to be run after `main_train.py` and with a `model/best_model.pt`.

The training script saves to `output/`, so that will have `best_model.pt`; just copy
it over to `model/` before running this.
"""
import numpy as np
import random
import data_processing
from torch_geometric.loader import DataLoader
import torch
from model import LearnedSimulator
import functional


if __name__ == '__main__':
    params = {
        'dataset_dir': 'data/tree_dataset/trial',
        'seed': 0,
        'model_weights': 'model/best_model.pt',
        'sim': False,
    }
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    # TODO(daniel): why are we not saving X_edges to a list?
    # This seems incorrect to me, need to fix how this script is used.
    X_force_list = []
    X_pos_list = []
    Y_pos_list = []
    for i in range(2,9):
        d = params['dataset_dir']+str(i)
        X_edges, X_force, X_pos, Y_pos = data_processing.load_npy(
                data_dir=d, sim=params['sim'], trial_num=i)
        X_force_list.append(X_force)
        X_pos_list.append(X_pos)
        Y_pos_list.append(Y_pos)
        print(f'On tree {i}, X_edges.T:\n{X_edges.T}')
    X_force_arr = np.concatenate(X_force_list)
    X_pos_arr = np.concatenate(X_pos_list)
    Y_pos_arr = np.concatenate(Y_pos_list)

    print('Loaded data, after concatenating:')
    print(f'  X_edges:     {X_edges.shape}')
    print(f'  X_force_arr: {X_force_arr.shape}')
    print(f'  X_pos_arr:   {X_pos_arr.shape}')
    print(f'  Y_pos_arr:   {Y_pos_arr.shape}')

    X_force_arr, X_pos_arr, Y_pos_arr = data_processing.shuffle_in_unison(
        X_force_arr, X_pos_arr, Y_pos_arr)

    train_val_split = int(len(X_force_arr)*0.9)

    X_force_train = X_force_arr[:train_val_split]
    X_pos_train = X_pos_arr[:train_val_split]
    Y_pos_train = Y_pos_arr[:train_val_split]

    X_force_val = X_force_arr[train_val_split:]
    X_pos_val = X_pos_arr[train_val_split:]
    Y_pos_val = Y_pos_arr[train_val_split:]

    val_dataset = data_processing.make_dataset(X_edges, X_force_val, X_pos_val, Y_pos_val,
                    make_directed=True, prune_augmented=False, rotate_augmented=False)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    print('Valid dataset size: {}'.format(len(val_dataset)))

    # NOTE(daniel): used to say GCN() for some reason, but we did LearnedSimulator().
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LearnedSimulator().to(device)
    model.load_state_dict(torch.load(params['model_weights']))
    functional.make_gif(model, test_loader, device)
