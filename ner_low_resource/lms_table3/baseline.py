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
parser.add_argument("--seed", default=10, type=int)
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

    filename = "ner_wikiann.pkl"

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

    target2pivot = {'cdo': 'zh',
                    'gn': 'es',
                    'ilo': 'id',
                    'mhr': 'ru',
                    'mi': 'id',
                    'tk': 'tr',
                    'qu': 'es',
                    'xmf': 'ka'}
    print(test_splits)
    for target_language in ['cdo', 'gn', 'ilo', 'mhr', 'mi', 'tk', 'qu', 'xmf']:
        label_f1 = args.label_f1.replace('**', target_language)

        pivot_language = target2pivot[target_language]

        target_test_f1 = []
        target_dev_f1 = []
        en_dev_f1 = []
        pivot_dev_f1 = []

        for idx in test_splits:
            en_dev_f1.append(data[idx]['dev_{}'.format('en')])
            target_dev_f1.append(data[idx]['dev_{}'.format(target_language)])
            target_test_f1.append(data[idx]['test_{}'.format(target_language)])
            pivot_dev_f1.append(data[idx]['dev_{}'.format(pivot_language)])

        en_select_idx = np.argmax(en_dev_f1)
        print(en_select_idx)
        en_select_res = target_test_f1[int(en_select_idx)]
        target_select_idx = np.argmax(target_dev_f1)
        target_select_res = target_test_f1[int(target_select_idx)]
        pivot_select_idx = np.argmax(pivot_dev_f1)
        pivot_select_res = target_test_f1[int(pivot_select_idx)]

        print(test_splits[pivot_select_idx])
        sort_target_test_f1 = list(np.sort(target_test_f1)[::-1])

        print('======{}======'.format(target_language))
        print('En-dev: {}----{}'.format(round(en_select_res*100, 1), sort_target_test_f1.index(en_select_res)))
        print('Pivot-dev: {}----{}'.format(round(pivot_select_res*100,1), sort_target_test_f1.index(pivot_select_res)))
        print('Target-dev: {}----{}'.format(round(target_select_res*100,1), sort_target_test_f1.index(target_select_res)))
        print('Target-test: {}'.format(round(np.max(target_test_f1)*100,1)))




