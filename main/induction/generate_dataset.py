#!/usr/bin/env python

"""Randomly generates dataset"""

import os
import argparse
import random
import re
import datetime
import sys
import json

from ntp.modules.generate import gen_simple, gen_relationships, write_data, write_relationships, write_simple_templates, gen_test_kb, gen_constant_dict, count_active
from ntp.util.util_data import load_conf
from ntp.util.util_kb import load_from_list, load_from_file, relationship_id_to_symbol

def write_list_to_file(list, path):
    """write relationship dict to file"""

    with open(path, "w") as f:
        for i in list: 
            f.write(i) 
    return

def write_dic_to_file(relationships, path):
    """write relationship dict to file"""

    with open(path, "w") as f:
        f.write(str(relationships))
    return

def _generate(conf):

    n_pred = conf["experiment"]["n_pred"]
    n_constants = conf["experiment"]["n_constants"]
    n_rel = conf["experiment"]["n_rel"]
    body_predicates = conf["experiment"]["n_body"]
    order = conf["experiment"]["order"]
    n_rules = conf["experiment"]["n_rules"]
    p_normal = conf["experiment"]["p_normal"]
    p_relationship = conf["experiment"]["p_relationship"]

    relationships = gen_relationships(n_pred, n_rel, body_predicates=body_predicates)
    symbol_relationships = relationship_id_to_symbol(relationships)

    train_data = gen_simple(n_pred, relationships, p_normal, p_relationship, n_constants, order=order)

    train_list = write_data(train_data)
    rules_list = write_simple_templates(n_rules, body_predicates=body_predicates, order=order)

    if conf["experiment"]["test"] == True:
        test_kb, train_list = gen_test_kb(train_list, conf["experiment"]["n_test"], conf["experiment"]["test_active_only"], relationships)
    else:
        test_kb = None            

    # print('-------------relationships-------------')
    # print(relationships)
    # print('-------------symbol_relationships-------------')
    # print(symbol_relationships)
    # print('-------------rules_list-------------')
    # print(rules_list)
    # print('-------------train list-------------')
    # print(train_list)
    # print('-------------test list-------------')
    # print(test_kb)

    exp_path = "C:\\Users\\giuseppe.pisano\\Documents\\MyProjects\\NSC4ExplainableAI\\LogicInduction\\NTP-improved\\data\\synthetic\\"
    write_list_to_file(rules_list, exp_path + conf['experiment']['name'] + '_templates.nlt')
    write_list_to_file(train_list, exp_path + conf['experiment']['name'] + '_train.nl')
    write_dic_to_file(relationships,  exp_path + conf['experiment']['name'] + '_relationships.nlr')


if __name__ == '__main__':
    conf = load_conf(sys.argv[1])
    _generate(conf)

    

    

    

    
