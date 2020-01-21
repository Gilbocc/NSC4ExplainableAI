import numpy as np
import torch
import torch.nn.functional as F
import sys
import dl2lib as dl2


def kl(p, log_p, log_q):
    return torch.sum(-p * log_q + p * log_p, dim=1)

def transform_network_output(o, network_output):
    if network_output == 'logits':
        pass
    elif network_output == 'prob':
        o = [F.softmax(zo) for zo in o]
    elif network_output == 'logprob':
        o = [F.log_softmax(zo) for zo in o]
    return o

class Constraint:

    def get_neighbor(self, x, y):
        assert False

    def get_condition(self, x, y):
        assert False

    def loss(self, x_batches, y_batches, args):
        constr = self.get_condition(x_batches, y_batches)
        
        neg_losses = dl2.Negate(constr).loss(args)
        pos_losses = constr.loss(args)
        sat = constr.satisfy(args)
        
        return neg_losses, pos_losses, sat