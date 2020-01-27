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
from training import run
import config


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


class AchilleConstraint(Constraint):

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


class CompleteConstraint(AchilleConstraint):

    def __init__(self, net, use_cuda=True, network_output='logits'):
        super().__init__(net, use_cuda, network_output)

    def get_condition(self, x, y):
        a = dl2.EQ(x[0], torch.tensor(self.names['Achille'], dtype=torch.float))
        b = dl2.EQ(x[0], torch.tensor(self.names['Pino'], dtype=torch.float))
        c = dl2.EQ(x[0], torch.tensor(self.names['Zenio'], dtype=torch.float))

        rules = [
            dl2.Implication(a, dl2.LT(y[0], y[1])),
            dl2.Implication(b, dl2.LT(y[1], y[0])),
            dl2.Implication(c, dl2.LT(y[0], y[1]))
        ]

        return dl2.And(rules)


def local_run(dataset_path, constraint_weight, global_constraining, num_epochs, random_seed, model_path, save_output):
    dataset = Values(dataset_path)
    model = Model()
    constraint = CompleteConstraint(model, use_cuda=False, network_output='logprob')
    oracle = DL2_Oracle(net=model, constraint=constraint, use_cuda=False)
    run(dataset, oracle, model, constraint_weight, global_constraining, num_epochs, random_seed, model_path, save_output)


parser = argparse.ArgumentParser(description='Experiment One')
parser = dl2.add_default_parser_args(parser)
args = parser.parse_args()
config.args = args

if __name__ == '__main__':
    # path = r'C:\Users\peppe_000\Documents\MyProjects\ExplainableAI\NetworkConstraining\DL2\main\dataset\output_simplified.csv'
    # model_path = r'C:\Users\peppe_000\Documents\MyProjects\ExplainableAI\NetworkConstraining\DL2\main\dataset\output_simplified_model_base.ph'
    # model_path = r'C:\Users\peppe_000\Documents\MyProjects\ExplainableAI\NetworkConstraining\DL2\main\dataset\output_simplified_model_constrained.ph'
    path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\University\NSC4ExplainableAI\NetworkConstraining\DL2\main\dataset\output_simplified.csv'
    # model_path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\University\NSC4ExplainableAI\NetworkConstraining\DL2\main\dataset\output_simplified_model_base.ph'
    model_path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\University\NSC4ExplainableAI\NetworkConstraining\DL2\main\dataset\output_simplified_model_constrained_complete.ph'
    save_output = True
    constraint_weight = 0.1
    global_constraining = True
    num_epochs = 10
    random_seed_base = 41
    num_runs = 1
    results = [local_run(path, constraint_weight, global_constraining, num_epochs, random_seed_base + i, model_path, save_output) for i in range(num_runs)]
    print('Mean accuracy for {:d} runs: {:.4f}'.format(num_runs, sum(results) / len(results)))