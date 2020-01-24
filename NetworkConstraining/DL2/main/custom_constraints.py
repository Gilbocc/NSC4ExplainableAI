import os
import torch
import dl2lib as dl2
import random
from common.constraint import Constraint

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