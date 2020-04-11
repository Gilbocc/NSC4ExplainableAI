import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import string
import numpy as np
import argparse
import pandas as pd

FEATURES = [str(x) for x in 'bcdefghijklmnopq']

FEATURE_MEANING = {
    'b' : 'HandicappedInfants',
    'c' : 'WaterProjectCostSharing',
    'd' : 'AdoptionOfTheBudgetResolution',
    'e' : 'PhysicianFeeFreeze',
    'f' : 'ElSalvadorAid',
    'g' : 'ReligiousGroupsInSchools',
    'h' : 'AntiSatelliteTestBan',
    'i' : 'AidToNicaraguanContras',
    'j' : 'MxMissile',
    'k' : 'Immigration',
    'l' : 'SynfuelsCorporationCutback',
    'm' : 'EducationSpending',
    'n' : 'SuperfundRightToSue',
    'o' : 'Crime',
    'p' : 'DutyFreeExports',
    'q' : 'ExportAdministrationActSouthAfrica',
}

class Model(nn.Module):
    def __init__(self, n_in=16, n_hidden=0, n_out=2):
        super(Model, self).__init__()
         
        self.linearlinear = nn.Sequential(
            nn.Linear(n_in, n_out, bias=True),
            nn.ReLU()
        )
        self.logprob = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.linearlinear(x)
        x = self.logprob(x)
        return x


def load_model(model_path):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    return model

def load_data(dataset_path):
        pd_data = pd.read_csv(dataset_path)
        return pd_data.iterrows()

def run(model_path, dataset_path, theory_path):
    model = load_model(model_path)
    data = load_data(dataset_path)
    theory = []
    republicans = []
    democrats = []
    theory.append('orientation(democrat).\n')
    theory.append('orientation(republican).\n')
    for index, elem in data:
        with torch.no_grad():
            parsed_elem = [elem[y] for y in FEATURES]
            data = Variable(torch.FloatTensor([parsed_elem]), requires_grad=False)
            c = model(data)
            y_val = np.argmax(c, axis=1)
            print(parsed_elem, ' ---> ', 'republican' if y_val.item() == 0 else 'democrat', y_val.item() == elem['a'])
            for feature in FEATURES:
                theory.append('{:s}{:s}({:s}).\n'.format('inFavourOf' if elem[feature] == 1 else 'contrarTo', FEATURE_MEANING[feature], 'republican' if y_val.item() == 0 else 'democrat'))

    # Salvare su file
    cleaned_theory = list(dict.fromkeys(theory))
    cleaned_theory.sort()
    with open(theory_path, 'w+') as theory_file:
        theory_file.writelines(cleaned_theory)
            
parser = argparse.ArgumentParser(description='Experiment two theory generator')
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--theory_path", type=str)
args = parser.parse_args()

if __name__ == '__main__':
    run(args.model_path, args.dataset_path, args.theory_path)
        