import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import argparse
import os

from utils import generate_from_labeled_file, generate_from_labeled_openstack_file
from model import Model


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


def create_dataloaders(normal_path, abnormal_path, window_size):
    normal_dataset = generate_from_labeled_file(normal_path, window_size)
    abnormal_dataset = generate_from_labeled_file(abnormal_path, window_size+1)

    size_normal = len(normal_dataset)
    size_abnormal = len(abnormal_dataset)

    # This split ratio is used to create a balanced test and
    # validation sets (equal number of normal and abnormal examples).
    normal_train_dataset, normal_test_val_dataset = \
        random_split(normal_dataset, [size_normal - size_abnormal, size_abnormal])

    test_split = 0.5
    val_split = 0.5

    test_size = int(size_abnormal * test_split / (test_split + val_split))
    val_size = size_abnormal - test_size

    normal_test_dataset, normal_val_dataset = random_split(normal_dataset, [test_size, val_size])
    abnormal_test_dataset, abnormal_val_dataset = random_split(abnormal_dataset, [test_size, val_size])

    normal_train_dl = DataLoader(normal_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    normal_test_dl = DataLoader(normal_test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    abnormal_test_dl = DataLoader(abnormal_test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    normal_val_dl = DataLoader(normal_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    abnormal_val_dl = DataLoader(abnormal_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return normal_train_dl, normal_test_dl, abnormal_test_dl, normal_val_dl, abnormal_val_dl

if __name__ == '__main__':

    # Hyperparameters
    num_classes = 31
    num_epochs = 310
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

    # Defining dataloaders
    normal_train_dl, normal_test_dl, abnormal_test_dl, normal_val_dl, abnormal_val_dl = \
        create_dataloaders('/home/toni/Downloads/balanced/normal_train.txt',
                           '/home/toni/Downloads/balanced/normal_train.txt',
                           window_size)

    writer = SummaryWriter(log_dir='log/' + log)

    model = Model(num_classes, hidden_size, num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001, threshold_mode='abs', cooldown=5, eps=0)

    # Train the model
    start_time = time.time()
    train_total_step = len(normal_train_dl)
    valid_total_step = len(normal_val_dl) + len(abnormal_val_dl)

    for epoch in range(num_epochs):  # Loop over the train and validation datasets multiple times
        train_loss = 0
        valid_loss = 0

        # Training
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

        # Cross-validation
        with torch.no_grad():

            # Normal validation dataset
            for step, (seq, label) in enumerate(normal_val_dl):
                seq = seq.clone().detach().view(-1, window_size).to(device)
                x_onehot = torch.nn.functional.one_hot(seq.long(), num_classes).float()
                output = model(x_onehot)
                loss = criterion(output, label.to(device))
                valid_loss += loss.item()

            # Abnormal validation dataset
            for step, (seq, label) in enumerate(abnormal_val_dl):
                seq = seq.clone().detach().view(-1, window_size).to(device)
                x_onehot = torch.nn.functional.one_hot(seq.long(), num_classes).float()
                output = model(x_onehot)


        print('Epoch [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / train_total_step, valid_loss / valid_total_step))
        writer.add_scalar('train_loss', train_loss / train_total_step, epoch + 1)
        writer.add_scalar('valid_loss', valid_loss / valid_total_step, epoch + 1)

        scheduler.step(valid_loss / valid_total_step)

        if epoch % snapshot_period == 0:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_dir + '/' + log + "_iteration=" + str(epoch) + '.pt')

        writer.flush()
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    writer.close()
    print('Finished Training')
