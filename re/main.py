import argparse
import numpy as np
import torch
from model import *
import lang2vec.lang2vec as l2v
import pickle
import random
# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--exp_name", default=None, type=str,
                    help="Checkpoint and config save prefix")
parser.add_argument("--batchsize", default=32, type=int)

parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=5, type=int)
parser.add_argument("--hidden_size", default=1024, type=int)
parser.add_argument("--seed", default=27, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--feature_task1", default='re_**_dev_repr', type=str, help='all, all_mono for all features togheter')
parser.add_argument("--feature_task2", default='0', type=str)
parser.add_argument("--target_task", default='re', type=str)
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

    if args.use_en == 0:
        lang_dic = {'arb': 0, 'zho': 1}
    else:
        lang_dic = {'arb': 0, 'zho': 1, 'eng': 2}

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

    model.to(device)
    model.set_device('cuda:' + args.gpuid)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    model = train(model, train_dataloader=train_dataloader, dev_dataloaders=dev_dataloaders,
                  optimizer=optimizer, max_epochs=args.max_epoch,
                  save_ckpt=save_ckpt, save_config=save_config)

    # Load best checkpoint

    output_model_file = save_ckpt
    model.load_state_dict(torch.load(output_model_file, map_location=device))


    best_model_f1, ndcg, kendall_score = evaluate(model, test_dataloader)
    print('TEST ACC: %.5f' % best_model_f1)
