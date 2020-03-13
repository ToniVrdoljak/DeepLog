import torch
from torch.utils.data import TensorDataset


def generate_from_labeled_hdfs_file(ds_path, window_size, num_classes):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(ds_path, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = list(map(int, line.strip().split()[2:]))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])

            if len(line) - window_size <= 0 < len(line):
                seq = line[:window_size + 1]
                seq += [num_classes] * (window_size + 1 - len(seq))
                inputs.append(seq[:-1])
                outputs.append(seq[-1])

    print('Number of sessions({}): {}'.format(ds_path, num_sessions))
    print('Number of seqs({}): {}'.format(ds_path, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


def generate_from_labeled_openstack_file(ds_path, window_size, num_classes):
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


def create_cross_val_loader(name, num_classes):
    hdfs = []
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(int, ln.strip().split()[2:]))
            ln = ln + [num_classes] * (window_size + 1 - len(ln))
            hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs
