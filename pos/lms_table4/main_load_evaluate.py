import argparse
import numpy as np
import torch
from model_multilingual_ft_feature import *
import pickle
import random
# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--exp_name", default='tb4-ft', type=str,
                    help="Checkpoint and config save prefix")
parser.add_argument("--batchsize", default=32, type=int)

parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=3, type=int)
parser.add_argument("--hidden_size", default=512, type=int)
parser.add_argument("--seed", default=8, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--target_language", default='zh', type=str)
parser.add_argument("--lang2vec", default='langrepr', type=str)


args = parser.parse_args()


if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_ckpt = args.exp_name + '.ckpt'
    save_config = args.exp_name + '.cfg'

    #print('Loading training data...\n')
    train_dataloader, dev_dataloaders, dev_target_dataloader, test_dataloader, proc = create_dataloader(batchsize=args.batchsize,
                                                                                 target_language=args.target_language,
                                                                                 )

    input_size = 512
    device = torch.device('cuda:' + args.gpuid)

    # TASK Language Dic

    lang_dic = {'arb': 0, 'deu': 1, 'spa': 2, 'nld': 3, 'zho': 4,
                'bul': 5, 'dan': 6, 'faa': 7, 'hun': 8, 'ita': 9,
                'por': 10, 'ron': 11, 'slk': 12, 'slv': 13, 'swe': 14}

    embedding = [[] for _ in range(len(lang_dic))]

    emb_size = None
    if args.lang2vec == 'langrepr':
        emb = np.load("lang_vecs.npy", allow_pickle=True, encoding='latin1')
        #print(emb.item().keys())
        for k, v in lang_dic.items():
            if k != 'eng':
                embedding[v] = emb.item()['optsrc' + k]
        emb_size = 512

    embedding = torch.FloatTensor(embedding)
    # MODEL
    model = RankNet(input_size=input_size, hidden_size=args.hidden_size, embedding=embedding, emb_dim=emb_size, activation='relu')

    model.set_device('cpu')

    # Load best checkpoint
    #print('Loading best check point...')
    output_model_file = save_ckpt
    model.load_state_dict(torch.load(output_model_file, map_location=device))

    #print('Evaluating on dev set...\n')
    best_model_f1, ndcg, kendall_score, rank = evaluate(model, test_dataloader)

    print('Target lang: %s, TEST ACC: %.5f' % (args.target_language, best_model_f1))
    # print('NDCG FINAL: %.5f' % ndcg)
    # print('KENDALL TEST: %.5f' % kendall_score)
    # print('TEST Rank: ', rank)
