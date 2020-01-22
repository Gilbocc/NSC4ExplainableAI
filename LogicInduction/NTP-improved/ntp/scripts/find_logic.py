#!/usr/bin/env python

"""Main experimental script. Randomly generates dataset and trains NTP model on dataset n times, and reports aggregate results."""

import os
import argparse
import random
import re
from collections import OrderedDict
import datetime
import sys

import tensorflow as tf
import numpy as np
from sklearn import metrics

from ntp.modules.train import train_model
from ntp.modules.eval import prop_rules, weighted_prop_rules, weighted_precision, confidence_accuracy
from ntp.modules.generate import gen_simple, gen_relationships, write_data, write_relationships, write_simple_templates, gen_test_kb, gen_constant_dict, count_active
from ntp.util.util_data import load_conf
from ntp.util.util_kb import load_from_list, load_from_file, relationship_id_to_symbol
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def read_rel_from_file(path):
    import ast
    with open(path, "r") as f:
        return ast.literal_eval(f.readline())

def read_list_from_file(path):
    with open(path, "r") as f:
        return f.readlines()

if __name__ == '__main__':

    tf.enable_eager_execution()

    print("GPUs Available: ", get_available_gpus())


    # parser = argparse.ArgumentParser()
    # parser.add_argument('-conf_path', default="conf_synth/algexp.conf")
    # args = parser.parse_args()

    path = sys.argv[1]

    conf = load_conf(path)

    n_pred = conf["experiment"]["n_pred"]
    n_constants = conf["experiment"]["n_constants"]
    n_rel = conf["experiment"]["n_rel"]
    body_predicates = conf["experiment"]["n_body"]
    order = conf["experiment"]["order"]
    n_rules = conf["experiment"]["n_rules"]
    p_normal = conf["experiment"]["p_normal"]
    p_relationship = conf["experiment"]["p_relationship"]
    base_seed = conf["experiment"]["base_seed"]

    random.seed(base_seed)
    np.random.seed(base_seed)

    n_runs = conf["experiment"]["n_runs"]

    base_dir = conf["logging"]["log_dir"] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    summary_writer = tf.contrib.summary.create_file_writer(base_dir + conf["experiment"]["name"])

    eval_history = OrderedDict()
    eval_keys = ["prop_rules", "active_facts"]

    if conf["experiment"]["test"] == True:
        eval_keys.extend(["MRR", "randomMRR", "fact-roc-auc"])

    for key in eval_keys:
        eval_history[key] = list()
    auc_helper_list = list()
    

    conf["logging"]["log_dir"] = base_dir + "run/"
    conf["training"]["seed"] = np.random.randint(100)

    relationships = read_rel_from_file(conf["data"]["rel_path"])
    symbol_relationships = relationship_id_to_symbol(relationships)

    train_list = read_list_from_file(conf["data"]["data_path"])
    rules_list = read_list_from_file(conf["data"]["rule_path"])

    # print(train_list)
    # print(rules_list)
    # print(symbol_relationships)

    if conf["experiment"]["test"] == True:
        test_kb, train_list = gen_test_kb(train_list, conf["experiment"]["n_test"], conf["experiment"]["test_active_only"], relationships)
    else:
        test_kb = None        

    kb = load_from_list(train_list)
    templates = load_from_list(rules_list, rule_template=True)

    rules, confidences, eval_dict = train_model(kb, templates, conf, relationships=symbol_relationships, test_kb=test_kb)
    
    constant_dict = gen_constant_dict(train_list)
    eval_dict["active_facts"] = count_active(constant_dict, relationships)
    eval_dict["active_ratio"] = eval_dict["active_facts"] / len(train_list)        
    
    print('-------------Statistics-------------')
    for key, value in eval_dict.items():
        if key in eval_history:
            print(key, value)
            eval_history[key].append(value)
    
    auc_helper_list.append(eval_dict["auc_helper"])
    print(eval_dict["auc_helper"])

    print('-------------Rules-------------')
    print(rules)
    print('-------------Confidences-------------')
    print(confidences)