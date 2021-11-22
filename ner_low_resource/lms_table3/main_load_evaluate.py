import argparse
import numpy as np
import torch
from model_evaluation import *
import pickle
import random
# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--exp_name", default='tmp', type=str,
                    help="Checkpoint and config save prefix")
parser.add_argument("--batchsize", default=32, type=int)

parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=3, type=int)
parser.add_argument("--hidden_size", default=512, type=int)
parser.add_argument("--seed", default=10, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--feature_task", default='0', type=str, help='all, all_mono for all features togheter')
parser.add_argument("--feature_task2", default='0', type=str, help='all, all_mono for all features togheter')
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
                                                                                 target_language=args.target_language
                                                                                 )

    input_size = 768
    device = torch.device('cuda:' + args.gpuid)

    # MODEL
    model = RankNet(input_size=input_size, hidden_size=args.hidden_size, activation='relu')

    model.to(device)
    model.set_device(device)

    # Load best checkpoint

    output_model_file = save_ckpt
    model.load_state_dict(torch.load(output_model_file, map_location=device))
    best_model_f1, ndcg, kendall_score, rank = evaluate(model, test_dataloader)
    print('Target language: %s TEST ACC: %.5f' % (args.target_language, best_model_f1))
