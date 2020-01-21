import numpy as np
import torch
import random

class DL2_Oracle:

    def __init__(self, net, constraint, use_cuda):
        self.net = net
        self.constraint = constraint
        self.use_cuda = use_cuda
        self.errors = []

    def evaluate(self, x_batches, y_batches, args):

        neg_losses_list = []
        pos_losses_list = []

        for i in range(0, len(x_batches[0]) - 1):
            x = x_batches[0][i]
            y = y_batches[0][i]
            neg_loss, pos_loss, _ =  self.constraint.loss(x, y, args)
            if pos_loss.item() != 0:
                self.errors.append((x, y, neg_loss, pos_loss))
            neg_losses_list.append(neg_loss)
            pos_losses_list.append(pos_loss)
        
        neg_losses = torch.stack([x for x in neg_losses_list])
        pos_losses = torch.stack([x for x in pos_losses_list])
        return torch.sum(neg_losses.float()), torch.sum(pos_losses.float()), 0

    def global_training(self, batch_size, num_epochs, optimize, data, target):
        for _ in range(num_epochs):
            for error in self.errors:
                generated = [self.constraint.get_neighbor(error[0], error[1], i) for i in range(batch_size // 4)] 
                original = [(data.data[i].float(), target.data[i].float()) for i in random.sample(range(0, len(data)), (batch_size // 4) * 3)]
                batch = generated + original
                random.shuffle(batch)
                batch_x = torch.stack([x[0] for x in batch])
                batch_y = torch.stack([x[1] for x in batch])
                optimize(batch_x, batch_y)
        self.errors = []