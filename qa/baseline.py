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
parser.add_argument("--seed", default=20, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--feature", default='0', type=str)
parser.add_argument("--label_f1", default='dev_**_f1', type=str)
parser.add_argument("--test_label_f1", default='test_**_f1', type=str)
parser.add_argument("--source_language", default='de-es-nl', type=str)
parser.add_argument("--target_language", default='zh', type=str)
parser.add_argument("--parallel_style", default='add', type=str)

args = parser.parse_args()

if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    filename = "qa_feature_all.pkl"

    print('Reading file: ', filename)

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

    target = sorted(['bg', 'da',  'hu', 'it', 'fa', 'pt', 'ro', 'sk', 'sl', 'sv'])
    target2pivot = {'bg': 'es',
             'da': 'nl',
                'hu': 'de', 'it': 'es', 'fa': 'es', 'pt': 'es', 'ro': 'es','sk': 'es','sl': 'es','sv': 'nl'}

    print('====================')
    lang2pivot = {'ar': 'de', 'de': 'ar', 'es': 'ar', 'zh': 'de', 'hi': 'es', 'vi': 'ar'}
    print('lang --- en-dev --- pivot-dev ')
    for lang in ['ar', 'de', 'es', 'zh', 'hi', 'vi']:
        total_en_dev_f1 = []
        total_target_dev_f1 = []
        total_target_test_f1 = []
        total_target_dev_100_f1 = []
        total_pivot_dev_f1 = []
        for i in test_splits:
            en_dev_f1 = data[i]['dev_en_f1']
            target_dev_f1 = data[i]['dev_**_f1'.replace('**', lang)]
            target_test_f1 = data[i]['test_**_f1'.replace('**', lang)]
            target_dev_100_f1 = data[i]['dev_**_f1_100'.replace('**', lang)]
            pivot_dev_f1 = data[i]['dev_**_f1'.replace('**', lang2pivot[lang])]
            total_en_dev_f1.append(en_dev_f1)
            total_target_dev_f1.append(target_dev_f1)
            total_target_test_f1.append(target_test_f1)
            total_target_dev_100_f1.append(target_dev_100_f1)
            total_pivot_dev_f1.append(pivot_dev_f1)

        #
        idx = np.argmax(total_en_dev_f1)
        predict = total_target_test_f1[idx]
        rank = list(np.sort(total_target_test_f1)[::-1]).index(predict)

        oracle_idx = np.argmax(total_target_dev_f1)
        dev_idx = np.argmax(total_target_dev_100_f1)
        p_idx = np.argmax(total_pivot_dev_f1)
        print('====================')
        print('{}--{}--{}'.format(lang, round(total_target_test_f1[idx],1), round(total_target_test_f1[p_idx], 1)))  #