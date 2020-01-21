import argparse
import os
import torch
import numpy as np
import torch.optim as optim
import json
import time
from torchvision import datasets, transforms
from common.oracle import DL2_Oracle
import sys
import dl2lib as dl2

import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from common.constraint import Constraint, transform_network_output

parser = argparse.ArgumentParser(description='Train NN with constraints')
parser = dl2.add_default_parser_args(parser)
args = parser.parse_args()

# We read the dataset and create an iterable.
class Values(data.Dataset):
    def __init__(self, filename):
        pd_data = pd.read_csv(filename)
        categorical_data = np.stack([pd_data[x].values for x in 'bcdefghijklmnopq'], 1)
        
        self.data = torch.tensor(categorical_data, dtype=torch.int64)
        self.target = torch.tensor(pd_data['a'].values).flatten()
        self.n_samples = self.data.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        b = self.data[index], self.target[index]
        # print(b)
        return b

class Model(nn.Module):
    def __init__(self, n_in=16, n_hidden=0, n_out=2):
        super(Model, self).__init__()
         
        self.linearlinear = nn.Sequential(
            nn.Linear(n_in, 10, bias=True),   # Hidden layer.
            nn.ReLU(),
            nn.Linear(10, n_out, bias=True),   # Hidden layer.
        )
        self.logprob = nn.LogSoftmax(dim=1) # -Log(Softmax probability).
    
    def forward(self, x):
        x = self.linearlinear(x)
        x = self.logprob(x)
        return x

class DummyConstraint(Constraint):

    def __init__(self, net, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 0
        self.name = 'DummyL'

    def params(self):
        return {'network_output' : self.network_output}

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        x_out = self.net(x_batches[0])
        x_out = transform_network_output([x_out], self.network_output)[0]
        x_out = x_out.detach()
        
        def evaluate():
            import random
            r = random.randint(0, len(x_batches[0]) - 1)
            x = x_batches[0][r]
            return x, x_out[r]

        x, y = evaluate()

        rules = [
            dl2.Implication(dl2.BoolConst(torch.FloatTensor([1, 0, 2]) == 1), dl2.LT(y[0], y[1])),
            dl2.Implication(dl2.BoolConst(torch.FloatTensor([1, 0, 2]) == 1), dl2.LT(y[1], y[0]))
        ]

        return dl2.And(rules)

def split_dataset(dataset, batch_size, validation_split, shuffle_dataset, random_seed):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader

def constrain(model, data, target):
    constraint = DummyConstraint(model, use_cuda=False, network_output='logprob')
    oracle = DL2_Oracle(learning_rate=0.01, net=model, constraint=constraint, use_cuda=False)

    n_batch = int(data.size()[0])
    x_batches, y_batches = [], []
    k = n_batch // oracle.constraint.n_tvars
    assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'
    for i in range(oracle.constraint.n_tvars):
        x_batches.append(data[i:(i + k)])
        y_batches.append(target[i:(i + k)])

    model.eval()

    if oracle.constraint.n_gvars > 0:
        domains = oracle.constraint.get_domains(x_batches, y_batches)
        z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=5, args=args)
        _, dl2_batch_loss, _ = oracle.evaluate(x_batches, y_batches, z_batches, args)
    else:
        _, dl2_batch_loss, _ = oracle.evaluate(x_batches, y_batches, None, args)

    model.train()

    return dl2_batch_loss

def train(train_loader, model, criterium, optimizer, constrain_weight):
    for k, (data, target) in enumerate(train_loader):
        # if (k == 20): return
        data = Variable(data.float(), requires_grad=False)
        target = Variable(target.long(), requires_grad=False)

        optimizer.zero_grad()
        pred = model(data)
        loss = criterium(pred, target)
        c_loss = constrain(model, data, target)
        print(c_loss)
        final_loss = loss + (c_loss + constrain_weight)
        final_loss.backward()
        optimizer.step()
        print('Loss {:.4f} at iter {:d}'.format(final_loss.item(), k))

def evaluate(validation_loader, model, criterium):
    
    res = [r for k, r in enumerate(validation_loader)]
    target = torch.cat([x[1] for x in res])
    data = torch.cat([x[0] for x in res])
    data = Variable(data.float(), requires_grad=False)
    target = Variable(target.long(), requires_grad=False)

    with torch.no_grad():
        y_val = model(data)
        loss = criterium(y_val, target)
        print(f'Loss: {loss:.8f}')

        y_val = np.argmax(y_val, axis=1)

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        print(confusion_matrix(target, y_val))
        print(classification_report(target, y_val))
        print(accuracy_score(target, y_val))
        return accuracy_score(target, y_val)

def run(dataset_path, constrain_weight):

    dataset = Values(dataset_path)

    train_loader, validation_loader = split_dataset(dataset, batch_size=10, validation_split=0.2, 
        shuffle_dataset=True, random_seed=41)
    
    model = Model()
    print(model)
    criterium = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train(train_loader, model, criterium, optimizer, constrain_weight)
    return evaluate(validation_loader, model, criterium)
            
if __name__ == '__main__':
    path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\NSC4ExplainableAI\NetworkConstraining\DL2\test\house-votes-84_parsed.csv'
    results = [run(path, 0.0) for i in range(100)]
    print('Mean accuracy for 100 runs')
    print(sum(results) / len(results))