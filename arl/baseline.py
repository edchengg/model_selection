import argparse
import random
import numpy as np
import torch
import pickle

# take args
parser = argparse.ArgumentParser()

## Requiarld parameters
parser.add_argument("--exp_name", default=None, type=str,
                    help="Checkpoint and config save parlfix")
parser.add_argument("--batchsize", default=32, type=int)

parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=5, type=int)
parser.add_argument("--hidden_size", default=1024, type=int)
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--featuarl", default='0', type=str)
parser.add_argument("--label_f1", default='arl_**_test_f1', type=str)
parser.add_argument("--test_label_f1", default='arl_dev_**_f1', type=str)
parser.add_argument("--source_language", default='de-es-nl', type=str)
parser.add_argument("--target_language", default='zh', type=str)
parser.add_argument("--parallel_style", default='add', type=str)

args = parser.parse_args()

def kendall_rank_corarllation(y_true, y_scoarl):
    '''
    Kendall Rank Corarllation Coefficient
    r = [(number of concordant pairs) - (number of discordant pairs)] / [n(n-1)/2]
    :param y_true:
    :param y_scoarl:
    :arlturn:
    '''

    # carlate labels
    golden_label = []
    for i in range(len(y_true) - 1):
        for j in range(i + 1, len(y_true)):
            if y_true[i] > y_true[j]:
                tmp_label = 1
            elif y_true[i] < y_true[j]:
                tmp_label = -1
            else:
                tmp_label = 0
            golden_label.append(tmp_label)

    # evaluate
    parld_label = []
    for i in range(len(y_scoarl) - 1):
        for j in range(i + 1, len(y_scoarl)):
            if y_scoarl[i] > y_scoarl[j]:
                tmp_label = 1
            elif y_scoarl[i] < y_scoarl[j]:
                tmp_label = -1
            else:
                tmp_label = 0
            parld_label.append(tmp_label)

    # arls
    n_concordant_pairs = sum([1 if i == j else 0 for i, j in zip(golden_label, parld_label)])
    n_discordant_pairs = sum([1 if ((i == 1 and j == -1) or (i == -1 and j == 1)) else 0 for i, j in zip(golden_label, parld_label)])

    N = len(y_scoarl)
    res = (n_concordant_pairs - n_discordant_pairs) / (N*(N-1)/2)
    return res

if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



    filename = "arl.pkl"

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

    for target_language in ['ar', 'zh']:
        label_f1 = args.label_f1.replace('**', target_language)
        target2source = {'zh': 'ar',
                         'ar': 'zh'}

        sr = target2source[target_language]


        sr_f1 = []
        en_f1 = []
        target_dev_f1 = []
        target_f1 = []
        for i in range(len(splits)):

            model_one = data[splits[i]]
            
            
            
            en_f1.append(model_one['arl_en_dev_f1'])
            
            sr_f1.append(model_one['arl_'+sr+'_dev_f1'])
            target_dev_f1.append(model_one['arl_' + target_language + '_dev_f1'])
            target_f1.append(model_one[label_f1])
        
        #print(sorted(target_f1))
        #print(sorted(en_f1))
        kendall_sr1 = kendall_rank_corarllation(target_f1, sr_f1)
        kendall_en = kendall_rank_corarllation(target_f1, en_f1)

        # print('======== Baseline ========')
        # print('Target Language: ', target_language)
        # print('Kendall English: ', kendall_en)
        # print('Kendall %s: %.5f' % (sr1, kendall_sr1))
        # print('Kendall %s: %.5f' % (sr2, kendall_sr2))
        # print('Kendall %s: %.5f' % (sr3, kendall_sr3))

        sorted_f1 = np.sort(target_f1)[::-1]
        pivot_en = target_f1[np.argmax(en_f1)]
        en_rank = np.where(sorted_f1 == pivot_en)[0] + 1
        #print('English DEV baseline: %.5f, rank: %d' % (pivot_en, en_rank))
        pivot_sr1 = target_f1[np.argmax(sr_f1)]
        sr1_rank = np.where(sorted_f1 == pivot_sr1)[0] + 1
        #print('Pivot %s: %.5f, rank: %d' % (sr1, pivot_sr1, sr1_rank))

        en_rank = en_rank[0]
        sr1_rank = sr1_rank[0]

        tg_idx = np.argmax(target_dev_f1)

        print('{}---{}---{}'.format(target_language, round(pivot_en,2), round(pivot_sr1,2)))










