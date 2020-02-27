import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import argparse
import os

from utils import generate_from_labeled_file


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


class Model(nn.Module):
    def __init__(self, num_keys, hidden_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_keys, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':

    # Hyperparameters
    num_classes = 31
    num_epochs = 300
    batch_size = 8192
    model_dir = 'model'
    log = 'Adam_batch_size={}_epochs={}'.format(str(batch_size), str(num_epochs))
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-snapshot_period', default=10, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    snapshot_period = args.snapshot_period

    model = Model(num_classes, hidden_size, num_layers).to(device)

    seq_dataset = generate_from_labeled_file('/home/toni/Downloads/train.txt', window_size)

    train_split = 0.8
    valid_split = 0.2

    train_size = int(len(seq_dataset) * train_split / (train_split + valid_split))
    valid_size = len(seq_dataset) - train_size

    train_dataset, valid_dataset = random_split(seq_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    writer = SummaryWriter(log_dir='log/' + log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    train_total_step = len(train_dataloader)
    valid_total_step = len(valid_dataloader)
    for epoch in range(num_epochs):  # Loop over the train and validation datasets multiple times
        train_loss = 0
        valid_loss = 0
        for step, (seq, label) in enumerate(train_dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size).to(device)
            x_onehot = torch.nn.functional.one_hot(seq.long(), num_classes).float()
            output = model(x_onehot)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        for step, (seq, label) in enumerate(valid_dataloader):
            seq = seq.clone().detach().view(-1, window_size).to(device)
            x_onehot = torch.nn.functional.one_hot(seq.long(), num_classes).float()
            output = model(x_onehot)
            loss = criterion(output, label.to(device))
            valid_loss += loss.item()
            #writer.add_graph(model, x_onehot)

        print('Epoch [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / train_total_step, valid_loss / valid_total_step))
        writer.add_scalar('train_loss', train_loss / train_total_step, epoch + 1)
        writer.add_scalar('valid_loss', valid_loss / valid_total_step, epoch + 1)

        if epoch % snapshot_period == 0:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_dir + '/' + log + "_iteration=" + str(epoch) + '.pt')

        writer.flush()
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    writer.close()
    print('Finished Training')
