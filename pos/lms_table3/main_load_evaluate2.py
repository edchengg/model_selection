import argparse
import numpy as np
import torch
from model2 import *
import lang2vec.lang2vec as l2v
import pickle
import random
# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--exp_name", default=None, type=str,
                    help="Checkpoint and config save prefix")
parser.add_argument("--batchsize", default=16, type=int)

parser.add_argument("--learning_rate", default=1e-5, type=float)
parser.add_argument("--max_epoch", default=5, type=int)
parser.add_argument("--hidden_size", default=512, type=int)
parser.add_argument("--seed", default=5, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--feature_task1", default='ud_**_dev_repr', type=str, help='all, all_mono for all features togheter')
parser.add_argument("--feature_task2", default='ud_**_dev_repr', type=str)
parser.add_argument("--target_task", default='pos', type=str)
parser.add_argument("--target_language", default='zh', type=str)
parser.add_argument("--activation", default='gelu', type=str)
parser.add_argument("--use_en", default=0, type=int)
parser.add_argument("--lang2vec", default='langrepr', type=str)


args = parser.parse_args()


if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_ckpt = args.exp_name + '.ckpt'
    save_config = args.exp_name + '.cfg'

    train_dataloader, dev_dataloaders, dev_target_dataloader, test_dataloader, proc = create_dataloader(batchsize=args.batchsize,
                                                                                 target_task=args.target_task,
                                                                                 target_language=args.target_language,
                                                                                 feature1=args.feature_task1,
                                                                                 feature2=args.feature_task2,
                                                                                 use_en=args.use_en)
    if 'mono' in args.feature_task1 or 'mono' in args.feature_task2:
        input_size = 768 * 3
    elif 'all' in args.feature_task1 or 'all' in args.feature_task2:
        input_size = 768 * 4
    else:
        input_size = 768

    device = torch.device('cuda:' + args.gpuid)

    # TASK Language Dic
    if args.target_task == 'conll':
        if args.use_en == 0:
            lang_dic = {'deu': 0, 'spa': 1, 'nld': 2, 'zho': 3}
        else:
            lang_dic = {'deu': 0, 'spa': 1, 'nld': 2, 'zho': 3, 'eng': 4}

        embedding = [[] for _ in range(len(lang_dic))]
    else:
        if args.use_en == 0:
            lang_dic = {'arb': 0, 'deu': 1, 'spa': 2, 'nld': 3, 'zho': 4,
                        'buk': 5, 'dan': 6, 'faa': 7, 'hun': 8, 'ita': 9, 'por': 10,
                        'ron': 11, 'slk': 12, 'slv': 13, 'swe': 14}
        else:
            lang_dic = {'arb': 0, 'deu': 1, 'spa': 2, 'nld': 3, 'zho': 4, 'eng': 5,
                        'buk': 6, 'dan': 7, 'faa': 8, 'hun': 9, 'ita': 10, 'por': 11,
                        'ron': 12, 'slk': 13, 'slv': 14, 'swe': 15
                        }
        embedding = [[] for _ in range(len(lang_dic))]

    # GET Language Representation
    emb_size = None
    if args.lang2vec == 'langrepr':
        emb = np.load("lang_vecs.npy", allow_pickle=True, encoding='latin1')
        for k, v in lang_dic.items():
            if k != 'eng':
                embedding[v] = emb.item()['optsrc' + k]
        emb_size = 512
    elif args.lang2vec == 'lang2vec':
        features = l2v.get_features(["eng", "deu", "spa", "nld", "zho", "arb"], "syntax_knn")
        for k, v in lang_dic.items():
            embedding[v] = features[k]
        emb_size = 103

    embedding = torch.FloatTensor(embedding)

    if 'all' in args.feature_task1 or 'all' in args.feature_task2:
        use_weight_sum = 1
    else:
        use_weight_sum = 0

    # MODEL
    model = RankNet(input_size=input_size, hidden_size=args.hidden_size, embedding=embedding, activation='relu',
                    emb_dim=emb_size, weighted_sum=use_weight_sum)


    output_model_file = save_ckpt
    model.load_state_dict(torch.load(output_model_file, map_location=device))
    model.set_device('cpu')


    best_model_f1, _, _ = evaluate(model, test_dataloader)
    print('Target lang: %s, TEST ACC: %.1f' % (args.target_language, round(best_model_f1*100, 1)))
