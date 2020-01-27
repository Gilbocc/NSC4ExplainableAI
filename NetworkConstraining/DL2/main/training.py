import os
import torch
import numpy as np
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import config

def split_dataset(dataset, batch_size, validation_split, shuffle_dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
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

    _, dl2_batch_loss, _ = oracle.evaluate(x_batches, y_batches, config.args)

    model.train()

    return dl2_batch_loss


def train(oracle, train_loader, model, criterium, optimizer, constraint_weight, global_constraining, num_epochs):

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

def run(dataset, oracle, model, constraint_weight, global_constraining, num_epochs, random_seed, model_path, save_output):

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_loader, validation_loader = split_dataset(dataset, batch_size=200, validation_split=0.2, shuffle_dataset=True)
    
    print(model)
    criterium = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for k in range(num_epochs):
        print('Training epoch number {:d}'.format(k))
        train(oracle, train_loader, model, criterium, optimizer, constraint_weight, global_constraining, num_epochs)

    if save_output:
        torch.save(model.state_dict(), model_path)

    return evaluate(validation_loader, model, criterium)