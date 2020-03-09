import torch
import torch.nn as nn
import time
import argparse

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    #hdfs = set()
    hdfs = []
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(int, ln.strip().split()[1:]))
            ln = ln #+ [-1] * (window_size + 1 - len(ln))
            #hdfs.add(tuple(ln))
            hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


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
    num_classes = 43
    model_path = "model/Adam_batch_size=64_epochs=3000_iteration=20.pt"
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = Model(num_classes, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = generate('/home/toni/Downloads/openstack/normal_test.txt')
    test_abnormal_loader = generate('/home/toni/Downloads/openstack/anomaly_representation.txt')
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in test_normal_loader:
            if len(line) < window_size:
                FP += 1
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
                x_onehot = torch.nn.functional.one_hot(seq.long(), num_classes).float()
                label = torch.tensor(label).view(-1).to(device)
                output = model(x_onehot)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break

    with torch.no_grad():
        for line in test_abnormal_loader:
            if len(line) < window_size:
                TP += 1
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size).to(device)
                x_onehot = torch.nn.functional.one_hot(seq.long(), num_classes).float()
                label = torch.tensor(label).view(-1).to(device)
                output = model(x_onehot)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)

    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
