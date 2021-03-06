import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from common.oracle import DL2_Oracle
import dl2lib as dl2
from common.constraint import Constraint
from common.training import run
import common.config as config


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

    def get_neighbor(self, x, y, index):
        pass

    def get_condition(self, x, y):

        rules = [
            dl2.Implication(dl2.BoolConst(torch.FloatTensor(1) == 1), dl2.LT(y[0], y[1])),
            dl2.Implication(dl2.BoolConst(torch.FloatTensor(1) == 1), dl2.LT(y[1], y[0]))
        ]

        return dl2.And(rules)


def local_run(dataset_path, constraint_weight, global_constraining, num_epochs, random_seed, model_path, save_output):
    dataset = Values(dataset_path)
    model = Model()
    constraint = DummyConstraint(model, use_cuda=False, network_output='logprob')
    oracle = DL2_Oracle(net=model, constraint=constraint, use_cuda=False)
    return run(dataset, oracle, model, constraint_weight, global_constraining, num_epochs, random_seed, model_path, save_output)


parser = argparse.ArgumentParser(description='Experiment votes')
parser = dl2.add_default_parser_args(parser)
args = parser.parse_args()
config.args = args
            
if __name__ == '__main__':
    path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\University\NSC4ExplainableAI\NetworkConstraining\DL2\main\dataset\experiment_votes\house-votes-84_parsed.csv'
    model_path = r''
    save_output = False
    constraint_weight = 0.0
    global_constraining = False
    num_epochs = 10
    random_seed_base = 41
    num_runs = 1
    results = [local_run(path, constraint_weight, global_constraining, num_epochs, random_seed_base + i, model_path, save_output) for i in range(num_runs)]
    print('Mean accuracy for {:d} runs: {:.4f}'.format(num_runs, sum(results) / len(results)))