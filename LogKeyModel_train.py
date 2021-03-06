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

from utils import generate_from_labeled_hdfs_file, generate_from_labeled_openstack_file, create_cross_val_loader
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


def create_train_dataloader(normal_train_path, window_size, num_classes):
    normal_train_dataset = generate_from_labeled_hdfs_file(normal_train_path, window_size, num_classes)
    normal_train_dataloader = DataLoader(normal_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return normal_train_dataloader


def train(model, optimizer, scheduler, train_dataloader, window_size):
    for epoch in range(num_epochs):  # Loop over the train and validation datasets multiple times
        train_loss = 0
        valid_loss = 0

        # Training
        for step, (seq, label) in enumerate(train_dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size).to(device)
            x_onehot = torch.nn.functional.one_hot(seq.long(), input_size).float()
            output = model(x_onehot)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / train_total_step))

        scheduler.step(train_loss / train_total_step)

        if epoch % snapshot_period == 0:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_dir + '/' + log + "_iteration=" + str(epoch) + '.pt')

    return train_loss


def validate(model, normal_val_loader, abnormal_val_loader, window_size):
    with torch.no_grad():
        TP = 0
        FP = 0

        # Normal validation dataset
        for line in normal_val_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
                x_onehot = torch.nn.functional.one_hot(seq.long(), input_size).float()
                label = torch.tensor(label).view(-1).to(device)
                output = model(x_onehot)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break

        # Abnormal validation dataset
        for line in abnormal_val_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
                x_onehot = torch.nn.functional.one_hot(seq.long(), input_size).float()
                label = torch.tensor(label).view(-1).to(device)
                output = model(x_onehot)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break

        FN = len(abnormal_val_loader) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
                FP, FN, P, R, F1))

        return P, R, F1


if __name__ == '__main__':

    # Hyperparameters
    num_classes = 31
    num_epochs = 1
    batch_size = 8192
    model_dir = 'model'
    log = 'cross_validation'
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-snapshot_period', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    num_candidates = args.num_candidates
    snapshot_period = args.snapshot_period

    window_size_list = [5, 6]#, 7, 8, 9, 10, 11]

    model_dict = dict()

    for window_size in window_size_list:

        normal_train_dataloader = create_train_dataloader('/home/toni/Downloads/balanced/normal_train.txt', window_size, num_classes)

        normal_val_loader = create_cross_val_loader('/home/toni/Downloads/balanced/normal_test.txt', window_size, num_classes)
        abnormal_val_loader = create_cross_val_loader('/home/toni/Downloads/balanced/anomaly.txt', window_size, num_classes)

        writer = SummaryWriter(log_dir='log/' + log)

        input_size = num_classes + 1

        model = Model(num_classes+1, hidden_size, num_layers, device).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001, threshold_mode='abs', cooldown=5, eps=0)

        # Train the model
        start_time = time.time()
        train_total_step = len(normal_train_dataloader)

        train_loss = train(model, optimizer, scheduler, normal_train_dataloader, window_size)

        P, R, F1 = validate(model, normal_val_loader, abnormal_val_loader, window_size)

        model_dict[window_size] = (model, P, R, F1, train_loss)
        writer.add_scalar('Precision', P, window_size)
        writer.add_scalar('Recall', R, window_size)
        writer.add_scalar('F1 score', F1, window_size)
        writer.add_scalar('train_loss', train_loss, window_size)

    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    writer.close()
    print('Finished Training')
