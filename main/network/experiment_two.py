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
        features = np.stack([pd_data[str(x)].values for x in range(1, 31)], 1)
        self.data = torch.tensor(features, dtype=torch.int64)
        self.target = torch.tensor(pd_data['Class'].values).flatten()
        self.n_samples = self.data.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        b = self.data[index], self.target[index]
        return b


class Model(nn.Module):
    def __init__(self, n_in=30, n_hidden=20, n_out=2):
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


class BaseConstraint(Constraint):

    def __init__(self, net, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 0
        self.name = 'BaseConstraint'

    def params(self):
        return {'network_output' : self.network_output}

    def get_neighbor(self, x, y, index):
        item = [random.randint(0, 1) for y in range(0, 30)]
        item[10] = x.data[10]
        # item[15] = x.data[15]
        # item[25] = x.data[25]
        item = torch.tensor(item)
        classification = torch.tensor(1, dtype=torch.float) if x.data[10] == 1 else torch.tensor(0, dtype=torch.float)
        return (item, classification)

    def get_condition(self, x, y):
        a = dl2.EQ(x[10], torch.tensor(1, dtype=torch.float))
        # b = dl2.EQ(x[15], torch.tensor(0, dtype=torch.float))
        # c = dl2.EQ(x[25], torch.tensor(1, dtype=torch.float))
        # condition = dl2.And([a, b, c])
        return dl2.Implication(a, dl2.LT(y[0], y[1]))


def local_run(dataset_path, constraint_weight, global_constraining, num_epochs, random_seed, model_path, save_output):
    dataset = Values(dataset_path)
    model = Model()
    constraint = BaseConstraint(model, use_cuda=False, network_output='logprob')
    oracle = DL2_Oracle(net=model, constraint=constraint, use_cuda=False)
    return run(dataset, oracle, model, constraint_weight, global_constraining, num_epochs, random_seed, model_path, save_output)


parser = argparse.ArgumentParser(description='Experiment Two')
parser = dl2.add_default_parser_args(parser)
parser.add_argument("--path", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--save_output", type=str)
parser.add_argument("--constraint_weight", type=str)
parser.add_argument("--global_constraining", type=str)
parser.add_argument("--num_epochs", type=str)
parser.add_argument("--random_seed_base", type=str)
parser.add_argument("--num_runs", type=str)
args = parser.parse_args()
config.args = args

if __name__ == '__main__':
    # path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\University\NSC4ExplainableAI\NetworkConstraining\DL2\main\dataset\experiment_two\dataset_final.csv'
    # model_path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\University\NSC4ExplainableAI\NetworkConstraining\DL2\main\dataset\experiment_two\dataset_model_final_no_constraining.ph'
    # save_output = True
    # constraint_weight = 0.0
    # global_constraining = True
    # num_epochs = 100
    # random_seed_base = 41
    # num_runs = 1
    path = args.path
    model_path = args.model_path
    save_output = args.save_output == 'True'
    constraint_weight = float(args.constraint_weight)
    global_constraining = args.global_constraining == 'True'
    num_epochs = int(args.num_epochs)
    random_seed_base = int(args.random_seed_base)
    num_runs = int(args.num_runs)
    results = [local_run(path, constraint_weight, global_constraining, num_epochs, random_seed_base + i, model_path, save_output) for i in range(num_runs)]
    print('Mean accuracy for {:d} runs: {:.4f}'.format(num_runs, sum(results) / len(results)))