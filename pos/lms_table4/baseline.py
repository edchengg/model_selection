import argparse
import random
import numpy as np
import torch
import pickle

# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--batchsize", default=32, type=int)

parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=5, type=int)
parser.add_argument("--hidden_size", default=1024, type=int)
parser.add_argument("--seed", default=8, type=int)
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

    filename = "pos_feature.pkl"



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
    print(test_splits)
    for target_language in target:
        label_f1 = args.label_f1.replace('**', target_language)

        target_test_f1 = []
        target_dev_f1 = []
        en_dev_f1 = []
        pivot_dev_f1 = []

        for idx in test_splits:
            en_dev_f1.append(data[idx]['{}_dev'.format('en')])
            target_dev_f1.append(data[idx]['{}_dev'.format(target_language)])
            target_test_f1.append(data[idx]['{}_test'.format(target_language)])
            pivot_dev_f1.append(data[idx]['{}_dev'.format(target2pivot[target_language])])

        en_select_idx = np.argmax(en_dev_f1)
        print(en_select_idx, test_splits[en_select_idx])

        en_select_res = target_test_f1[int(en_select_idx)]
        target_select_idx = np.argmax(target_dev_f1)
        target_select_res = target_test_f1[int(target_select_idx)]


        pivot_select_idx = np.argmax(pivot_dev_f1)
        pivot_select_res = target_test_f1[int(pivot_select_idx)]

        print(test_splits[pivot_select_idx])
        sort_target_test_f1 = list(np.sort(target_test_f1)[::-1])

        print('{}--{}--{}'.format(round(en_select_res*100, 1),round(pivot_select_res*100,1)))




