import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import string
import numpy as np

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

def random_strings(number, max_length):
    def random_string(length):
        return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

    return [random_string(random.randint(4, max_length)).capitalize() for x in range (0, number)]

def generate_data():
    num_surnames = 100
    num_names = 3
    surnames = random_strings(num_surnames, 10)
    names = ['Pino', 'Zenio', 'Achille']
    
    people = [(names[random.randint(0, num_names - 1)], surnames[random.randint(0, num_surnames - 1)]) for x in range(100)]
    people = list(map(lambda x : (x[0], x[1]), people))
    return people

def load_model(model_path):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    return model

def run(model_path, theory_path):
    model = load_model(model_path)
    names = {
        'Achille' : 0,
        'Pino' : 1,
        'Zenio' : 2
    }
    for elem in generate_data():
        with torch.no_grad():
            data = Variable(torch.FloatTensor([[names[elem[0]], random.randint(0, 1000)]]), requires_grad=False)
            c = model(data)
            y_val = np.argmax(c, axis=1)
            print('Predicted value for elem ({:s}, {:s}) is {:d}'.format(elem[0], elem[1], y_val.item()))
            # Salvare su file

if __name__ == '__main__':
    model_path = r'C:\Users\peppe_000\Documents\MyProjects\ExplainableAI\NetworkConstraining\DL2\main\dataset\output_simplified_model.ph'
    theory_path = r''
    run(model_path, theory_path)