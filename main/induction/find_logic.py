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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description='Rules finder')
parser.add_argument("--config_path", type=str)
parser.add_argument("--theory_path", type=str)
parser.add_argument("--rules_template_path", type=str)
parser.add_argument("--rules_path", type=str)
args = parser.parse_args()


if __name__ == '__main__':

    tf.test.is_built_with_cuda()
    tf.enable_eager_execution()
    # print("GPUs Available: ", get_available_gpus())

    conf = load_conf(args.config_path)
    conf["data"]["data_path"] = args.theory_path
    conf["data"]["rule_path"] = args.rules_template_path
    conf["data"]["target_path"] = args.rules_path

    base_seed = conf["experiment"]["base_seed"]
    random.seed(base_seed)
    np.random.seed(base_seed)

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
    conf["training"]["seed"] = base_seed

    if conf["data"]["rel_path"] != '':
        relationships = read_rel_from_file(conf["data"]["rel_path"])
        symbol_relationships = relationship_id_to_symbol(relationships)
    else:
        symbol_relationships = None

    train_list = read_list_from_file(conf["data"]["data_path"])
    rules_list = read_list_from_file(conf["data"]["rule_path"])

    if conf["experiment"]["test"] == True:
        test_kb, train_list = gen_test_kb(train_list, conf["experiment"]["n_test"], conf["experiment"]["test_active_only"], relationships)
    else:
        test_kb = None        

    kb = load_from_list(train_list)
    templates = load_from_list(rules_list, rule_template=True)

    rules, confidences, stringified_rules, eval_dict = train_model(kb, templates, conf, relationships=symbol_relationships, test_kb=test_kb)
    
    print('-------------Rules-------------')
    print(stringified_rules)
    print('-------------Confidences-------------')
    print(confidences)

    # Saving results
    with open(conf["data"]["target_path"], 'w+') as target_file:
        stringified_rules = list(map(lambda x, y : x + '    ' + str(y) + '\n', stringified_rules, confidences))
        target_file.writelines(list(dict.fromkeys(stringified_rules)))