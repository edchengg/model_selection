import argparse
import random
import numpy as np
import torch
from model import *
import pickle

# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--batchsize", default=32, type=int)

parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=5, type=int)
parser.add_argument("--hidden_size", default=1024, type=int)
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--feature", default='0', type=str)
parser.add_argument("--label_f1", default='conll_dev_**_f1', type=str)
parser.add_argument("--test_label_f1", default='conll_dev_**_f1', type=str)
parser.add_argument("--source_language", default='de-es-nl', type=str)
parser.add_argument("--target_language", default='zh', type=str)
parser.add_argument("--parallel_style", default='add', type=str)

args = parser.parse_args()


if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    filename =  "ner.pkl"

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    length = len(data)
    splits = []
    for k, v in data.items():
        splits.append(k)

    random.shuffle(splits)

    train_splits = splits[:int(0.5 * length)]
    dev_splits = splits[int(0.5 * length):int(0.75 * length)]
    test_splits = splits[int(0.75 * length):]
    splits = test_splits
    oracle = []
    en_dev_f1 = []
    en_dev_rank = []
    en_dev_kendall = []
    pivot_dev_f1 = []
    pivot_dev_rank = []
    pivot_dev_kendall = []
    lang2pivot = {'de':'nl', 'es':'nl','nl':'de','zh':'de'}
    for target_language in ['de', 'es', 'nl', 'zh']:

        total_en_dev_f1 = []
        total_target_dev_f1 = []
        total_target_test_f1 = []
        total_target_dev_100_f1 = []
        total_pivot_dev_f1 = []
        for i in test_splits:
            en_dev_f1 = data[i]['conll_dev_f1']
            target_test_f1 = data[i]['conll_test_**_f1'.replace('**', target_language)]
            pivot_dev_f1 = data[i]['conll_dev_**_f1'.replace('**', lang2pivot[target_language])]
            total_en_dev_f1.append(en_dev_f1* 100)
            total_target_test_f1.append(target_test_f1* 100)
            total_pivot_dev_f1.append(pivot_dev_f1* 100)


        idx = np.argmax(total_en_dev_f1)
        predict = total_target_test_f1[idx]
        rank = list(np.sort(total_target_test_f1)[::-1]).index(predict)
        p_idx = np.argmax(total_pivot_dev_f1)

        print('{}--{}--{}'.format(target_language, round(total_target_test_f1[idx], 1),
                                      round(total_target_test_f1[p_idx], 1)))