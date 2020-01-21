import argparse
import os
import torch
import numpy as np
import torch.optim as optim
import json
import time
from torchvision import datasets, transforms
from oracles import DL2_Oracle
import sys
import dl2lib as dl2
import random

import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from constraints import Constraint, transform_network_output

parser = argparse.ArgumentParser(description='Train NN with constraints')
parser = dl2.add_default_parser_args(parser)
args = parser.parse_args()

# We read the dataset and create an iterable.
class Values(data.Dataset):
    def __init__(self, filename):
        pd_data = pd.read_csv(filename)
        categorical_columns = ['Name', 'Surname']
        for category in categorical_columns:
            pd_data[category] = pd_data[category].astype('category')
        print(pd_data['Name'].cat.categories)
        names = pd_data['Name'].cat.codes.values.astype('int64')
        surnames = pd_data['Surname'].cat.codes.values.astype('int64')
        categorical_data = np.stack([names, surnames], 1)
        
        self.data = torch.tensor(categorical_data, dtype=torch.int64)
        self.target = torch.tensor(pd_data['Class'].values).flatten()
        self.n_samples = self.data.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        b = self.data[index], self.target[index]
        return b

class Model(nn.Module):
    def __init__(self, n_in=2, n_hidden=20, n_out=2):
        super(Model, self).__init__()
         
        self.linearlinear = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out, bias=True),
        )
        self.logprob = nn.LogSoftmax(dim=1)
    
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
        self.names = {
            'Achille' : 0,
            'Pino' : 1,
            'Zenio' : 2
        }

    def params(self):
        return {'network_output' : self.network_output}

    def get_neighbor(self, x, y, index):
        # item = torch.tensor([x.data[0], x.data[1] + index])
        item = torch.tensor([x.data[0], random.randint(0, 100000)])
        classification = torch.tensor(1, dtype=torch.float) if x.data[0] == self.names['Achille'] or x.data[0] == self.names['Zenio'] else torch.tensor(0, dtype=torch.float)
        return (item, classification)

    def get_condition(self, x, y):
        a = dl2.EQ(x[0], torch.tensor(self.names['Achille'], dtype=torch.float))
        return dl2.Implication(a, dl2.LT(y[0], y[1]))

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

def local_constraining(oracle, model, data, target):

    n_batch = int(data.size()[0])
    x_batches, y_batches = [], []
    k = n_batch // oracle.constraint.n_tvars

    assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'
    for i in range(oracle.constraint.n_tvars):
        x_batches.append(data[i:(i + k)])
        y_batches.append(target[i:(i + k)])

    model.eval()

    _, dl2_batch_loss, _ = oracle.evaluate(x_batches, y_batches, args)

    model.train()

    return dl2_batch_loss

def train(train_loader, model, criterium, optimizer, constraint_weight, global_constraining, num_epochs):

    constraint = DummyConstraint(model, use_cuda=False, network_output='logprob')
    oracle = DL2_Oracle(net=model, constraint=constraint, use_cuda=False)

    for k, (data, target) in enumerate(train_loader):
        data_variable = Variable(data.float(), requires_grad=False)
        target_variable = Variable(target.long(), requires_grad=False)

        optimizer.zero_grad()

        # Traditional loss
        pred = model(data_variable)
        loss = criterium(pred, target_variable)

        # Using predictions to adjust the loss (constraining based on train data)
        c_loss = local_constraining(oracle, model, data_variable, pred)
        print('Constraint Loss {:.4f} at iter {:d}'.format(c_loss.item(), k))
        final_loss = loss + (c_loss * constraint_weight)

        # Computing gradient and updating weights
        final_loss.backward()
        optimizer.step()

        print('Loss {:.4f} at iter {:d}'.format(final_loss.item(), k))

        # Global constraining
        def optimize(x_batches, y_batches):
            data = Variable(x_batches.float(), requires_grad=False)
            target = Variable(y_batches.long(), requires_grad=False)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterium(pred, target)
            print('Ghost Loss {:.4f} at iter {:d}'.format(loss.item(), k))
            loss.backward()
            optimizer.step()

        if global_constraining:
            oracle.global_training(200, num_epochs, optimize, data, target)

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
        print('Accuracy score: {:.4f}'.format(accuracy_score(target, y_val)))
        return accuracy_score(target, y_val)

def run(dataset_path, constraint_weight, global_constraining, num_epochs, random_seed):

    dataset = Values(dataset_path)

    train_loader, validation_loader = split_dataset(dataset, batch_size=200, validation_split=0.2, 
        shuffle_dataset=True, random_seed=random_seed)
    
    model = Model()
    print(model)
    criterium = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for k in range(num_epochs):
        print('Training epoch number {:d}'.format(k))
        train(train_loader, model, criterium, optimizer, constraint_weight, global_constraining, num_epochs)

    return evaluate(validation_loader, model, criterium)
            
if __name__ == '__main__':
    path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\NSC4ExplainableAI\NetworkConstraining\DL2\test\output_simplified_2.csv'
    constraint_weight = 0.0
    global_constraining = False
    num_epochs = 50
    random_seed_base = 41
    num_runs = 1
    results = [run(path, constraint_weight, global_constraining, num_epochs, random_seed_base + i) for i in range(num_runs)]
    print('Mean accuracy for {:d} runs: {:.4f}'.format(num_runs, sum(results) / len(results)))