import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import string
import numpy as np
import argparse
import pandas as pd

FEATURES = [str(x) for x in range(1, 31)]
NUM_ROWS = 100


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


def generate_data():
    data = [[random.randint(0, 1) for y in FEATURES] for _ in range (0, NUM_ROWS)]
    return data


def load_model(model_path):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    return model


def run_from_model(model_path, theory_path):
    model = load_model(model_path)
    ok = ['ok(true).\n'] + ['has{:s}({:s}) :- ok({:s}).\n'.format(y, ','.join(['X' + x for x in FEATURES]), 'X' + y) for y in FEATURES]
    notOk = ['notOk(false).\n'] + ['hasNot{:s}({:s}) :- notOk({:s}).\n'.format(y, ','.join(['X' + x for x in FEATURES]), 'X' + y) for y in FEATURES]
    theory = ok + notOk
    for elem in generate_data():
        with torch.no_grad():
            data = Variable(torch.FloatTensor([elem]), requires_grad=False)
            c = model(data)
            y_val = np.argmax(c, axis=1)
            print(elem, ' ---> ', y_val.item(), elem[10] == 1)
            if y_val.item() == 1:
                theory.append('isA({:s}).\n'.format(','.join(map(lambda x: 'false' if x == 0 else 'true', elem))))
            else:
                theory.append('isB({:s}).\n'.format(','.join(map(lambda x: 'false' if x == 0 else 'true', elem))))

    # Salvare su file
    with open(theory_path, 'w+') as theory_file:
        theory_file.writelines(list(dict.fromkeys(theory)))

def run_from_csv(dataset_path, theory_path):

    def load_data():
        pd_data = pd.read_csv(dataset_path)
        return pd_data.iterrows()

    ok = ['ok(true).\n'] + ['has{:s}({:s}) :- ok({:s}).\n'.format(y, ','.join(['X' + x for x in FEATURES]), 'X' + y) for y in FEATURES]
    notOk = ['notOk(false).\n'] + ['hasNot{:s}({:s}) :- notOk({:s}).\n'.format(y, ','.join(['X' + x for x in FEATURES]), 'X' + y) for y in FEATURES]
    theory = ok + notOk
    for index, elem in load_data():
        data = [elem[y] for y in FEATURES]
        if elem['Class'] == 1:
                theory.append('isA({:s}).\n'.format(','.join(map(lambda x: 'false' if x == 0 else 'true', data))))
        else:
            theory.append('isB({:s}).\n'.format(','.join(map(lambda x: 'false' if x == 0 else 'true', data))))


    # Salvare su file
    with open(theory_path, 'w+') as theory_file:
        theory_file.writelines(list(dict.fromkeys(theory)))
            
parser = argparse.ArgumentParser(description='Experiment two theory generator')
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--theory_path", type=str)
parser.add_argument("--is_model", type=str)
args = parser.parse_args()

if __name__ == '__main__':
    if args.is_model == 'True':
        run_from_model(args.model_path, args.theory_path)
    else:
        run_from_csv(args.dataset_path, args.theory_path)