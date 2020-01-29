import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import string
import numpy as np

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


def run(model_path, theory_path):
    model = load_model(model_path)
    
    theory = []
    for elem in generate_data():
        with torch.no_grad():
            data = Variable(torch.FloatTensor([elem]), requires_grad=False)
            c = model(data)
            y_val = np.argmax(c, axis=1)
            print(elem, ' ---> ', y_val.item(), elem[10] == 1 and elem[15] == 0 and elem[25] == 1)
            # Convert result to prolog
            theory.append('class({:d}).\n'.format(y_val.item()))
            theory.append('name({:s}).\n'.format(elem[0].lower()))
            theory.append('surname({:s}).\n'.format(elem[1].lower()))
            theory.append('nameWithClass({:s},{:d}).\n'.format(elem[0].lower(), y_val.item()))
            theory.append('surnameWithClass({:s},{:d}).\n'.format(elem[1].lower(), y_val.item()))
            theory.append('person({:s},{:s}).\n'.format(elem[0].lower(), elem[1].lower()))
            theory.append('classification({:s},{:s},{:d}).\n'.format(elem[0].lower(), elem[1].lower(), y_val.item()))

    # Salvare su file
    # with open(theory_path, 'w+') as theory_file:
    #     theory_file.writelines(list(dict.fromkeys(theory)))
            

if __name__ == '__main__':
    model_path = r'C:\Users\giuseppe.pisano\Documents\MyProjects\University\NSC4ExplainableAI\NetworkConstraining\DL2\main\dataset\experiment_two\dataset_model_base.ph'
    theory_path = r''
    run(model_path, theory_path)