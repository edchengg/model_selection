import os
import argparse
import random
import numpy as np
import torch
from model import *
from transformers import *
import pickle
# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--source_language", default='en', type=str,
                    help="The target language")
parser.add_argument("--target_language", default='en', type=str,
                    help="The target language")
parser.add_argument("--bert_model", default='bert-base-multilingual-cased', type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")

parser.add_argument("--output_dir", default='save', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--ckpt", default=None, type=str,
                    help="Checkpoint for previously saved mdoel")
parser.add_argument("--exp_name", default='0', type=str,
                    help="Checkpoint and config save prefix")
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument("--num_exp", default=None, type=int,
                    help="Number of additional examples from source language")
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=5, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--num_duplicate", default=20, type=int)
parser.add_argument("--warmup_proportion", default=0.4, type=float)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
args = parser.parse_args()


if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_ckpt = '/srv/share4/ychen3411/project00_model_save/pos/{}'.format(args.exp_name) + '/' + args.exp_name + '.ckpt'
    save_config = '/srv/share4/ychen3411/project00_model_save/pos/{}'.format(args.exp_name) + '/'  + args.exp_name + '.cfg'

    if not os.path.exists('/srv/share4/ychen3411/project00_model_save/pos/{}'.format(args.exp_name)):
        os.mkdir('/srv/share4/ychen3411/project00_model_save/pos/{}'.format(args.exp_name))

    # parse source domains
    print('F1 ================== EXP =====================')
    source_language = args.source_language
    target_language = args.target_language
    print('F1 Target language: %s' % target_language)

    print('batchsize: %d' % args.batchsize)
    print('learning rate: %.7f' % args.learning_rate)
    print('max epochs: %d' % args.max_epoch)
    print('max_seq_length: %d' % args.max_seq_length)
    print('num_depulicate: %d' % args.num_duplicate)
    print('warmup proportion: %.5f' % args.warmup_proportion)
    print('model ckpt will be saved at: %s' % save_ckpt)
    print('model config will be saved at: %s' % save_config)

    processor = UDProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    device = torch.device('cuda:' + args.gpuid)
    # build model
    if args.bert_model == 'bert-base-multilingual-cased':
        model = BertForPOS.from_pretrained(args.bert_model,
                                    cache_dir=args.output_dir,
                                    num_labels = num_labels,
                                    output_hidden_states=True) # if you want to get all layer hidden states
    elif args.bert_model == 'xlm-roberta-base':
        model = XLMRobertaForPOS.from_pretrained(args.bert_model,
                                           cache_dir=args.output_dir,
                                           num_labels=num_labels,
                                           output_hidden_states=True)  # if you want to get all layer hidden states
    elif args.bert_model == 'xlm-mlm-xnli15-1024':
        model = XLMForPOS.from_pretrained(args.bert_model,
                                           cache_dir=args.output_dir,
                                           num_labels=num_labels,
                                           output_hidden_states=True)  # if you want to get all layer hidden states
    elif args.bert_model == 'xlm-mlm-tlm-xnli15-1024':
        model = XLMForPOS.from_pretrained(args.bert_model,
                                           cache_dir=args.output_dir,
                                           num_labels=num_labels,
                                           output_hidden_states=True)  # if you want to get all layer hidden states
    elif args.bert_model == 'xlm-roberta-large':
        model = XLMRobertaForPOS.from_pretrained(args.bert_model,
                                           cache_dir=args.output_dir,
                                           num_labels=num_labels,
                                           output_hidden_states=True)  # if you want to get all layer hidden states
    else:
        config = BertConfig.from_json_file('/home/chen.9279/MBERT/bibert/bert_config.json')
        config.num_labels = num_labels
        config.output_hidden_states = True
        model = BertForPOS(config=config)
        model.load_state_dict(torch.load('/home/chen.9279/MBERT/bibert/pytorch_model.ckpt', map_location=device),
                              strict=False)

    model.set_label_map(label_list)

    model.to(device)
    model.set_device('cuda:' + args.gpuid)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    lang2full = {'zh': 'Chinese',
                 'ar': 'Arabic',
                 'bg': 'Bulgarian',
                 'da': 'Danish',
                 'nl': 'Dutch',
                 'en': 'English',
                 'de': 'German',
                 'hu': 'Hungarian',
                 'it': 'Italian',
                 'fa': 'Persian',
                 'pl': 'Polish',
                 'pt': 'Portuguese',
                 'ro': 'Romanian',
                 'sk': 'Slovak',
                 'sl': 'Slovenian',
                 'es': 'Spanish',
                 'sv': 'Swedish'}

    if args.bert_model == 'bert-base-multilingual-cased':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        tokenizer.bos_token = '[CLS]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.unk_token = '[UNK]'
        tokenizer.sep_token = '[SEP]'
        tokenizer.cls_token = '[CLS]'
        tokenizer.mask_token = '[MASK]'
        tokenizer.pad_token = '[PAD]'
    elif args.bert_model == 'xlm-roberta-base':
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    elif args.bert_model == 'xlm-roberta-large':
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    elif args.bert_model == 'xlm-mlm-xnli15-1024':
        tokenizer = XLMTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        tokenizer.bos_token = '<s>'
        tokenizer.eos_token = '</s>'
        tokenizer.unk_token = '<unk>'
        tokenizer.sep_token = '</s>'
        tokenizer.cls_token = '</s>'
        tokenizer.mask_token = '<special1>'
        tokenizer.pad_token = '<pad>'

    elif args.bert_model == 'xlm-mlm-tlm-xnli15-1024':
        tokenizer = XLMTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        tokenizer.bos_token = '<s>'
        tokenizer.eos_token = '</s>'
        tokenizer.unk_token = '<unk>'
        tokenizer.sep_token = '</s>'
        tokenizer.cls_token = '</s>'
        tokenizer.mask_token = '<special1>'
        tokenizer.pad_token = '<pad>'
    else:
        tokenizer = BertTokenizer.from_pretrained('/home/chen.9279/MBERT/bibert/vocab.txt', do_lower_case=False)
        tokenizer.bos_token = '[CLS]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.unk_token = '[UNK]'
        tokenizer.sep_token = '[SEP]'
        tokenizer.cls_token = '[CLS]'
        tokenizer.mask_token = '[MASK]'
        tokenizer.pad_token = '[PAD]'


    # Load best checkpoint
    # print('Loading best check point...')
    output_model_file = save_ckpt
    model.load_state_dict(torch.load(output_model_file, map_location=device))

    save_dir = 'evaluation/' + args.exp_name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for target_language in list(lang2full.keys()):
        dev_data_path = 'UD_' + lang2full[target_language] + '/' + target_language + '-ud-dev.conllu'
        test_data_path = 'UD_' + lang2full[target_language] + '/' + target_language + '-ud-test.conllu'
        print('Loading development data...\n')
        dev_dataloader, dev_size = create_dataloader(dev_data_path, set_type='dev',
                                                     batchsize=args.batchsize,
                                                     max_seq_length=args.max_seq_length, tokenizer=tokenizer,
                                                     num_duplicate=args.num_duplicate)
        print('Loading testing data...\n')
        test_dataloader, test_size = create_dataloader(test_data_path, set_type='test',
                                                       batchsize=args.batchsize,
                                                       max_seq_length=args.max_seq_length, tokenizer=tokenizer,
                                                       num_duplicate=args.num_duplicate)


        # test
        print('Evaluating on dev set...\n')
        dev_acc, _, dev_gold, dev_pred = evaluate_save(model, dev_dataloader, dev_size)

        print('Evaluating on test set...\n')
        test_acc, _, test_gold, test_pred = evaluate_save(model, test_dataloader, test_size)

        dic = {'test_gold': test_gold,
         'test_pred': test_pred}
        with open(save_dir + '/' + target_language + '.pkl', 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(save_dir + '/{}.txt'.format(target_language), 'w') as f:
            f.write('dev={}\n'.format(dev_acc))
            f.write('test={}\n'.format(test_acc))

        rep = extract_features(model, dev_dataloader, dev_size)

        np.save(save_dir + '/' + target_language, rep)
