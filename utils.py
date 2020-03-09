import torch
from torch.utils.data import TensorDataset


def generate_from_labeled_file(ds_path, window_size):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(ds_path, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(int, line.strip().split()[2:]))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(ds_path, num_sessions))
    print('Number of seqs({}): {}'.format(ds_path, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


def generate_from_labeled_openstack_file(ds_path, window_size):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(ds_path, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(int, line.strip().split()[1:]))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(ds_path, num_sessions))
    print('Number of seqs({}): {}'.format(ds_path, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset
