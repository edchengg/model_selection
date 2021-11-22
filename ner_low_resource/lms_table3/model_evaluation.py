import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
import pickle
import numpy as np


def dcg_score(y_true, y_score, k=30, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1

    discounts = np.log2(np.arange(len(y_true)) + 2)

    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=30, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def kendall_rank_correlation(y_true, y_score):
    '''
    Kendall Rank Correlation Coefficient
    r = [(number of concordant pairs) - (number of discordant pairs)] / [n(n-1)/2]
    :param y_true:
    :param y_score:
    :return:
    '''

    # create labels
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
    pred_label = []
    for i in range(len(y_score) - 1):
        for j in range(i + 1, len(y_score)):
            if y_score[i] > y_score[j]:
                tmp_label = 1
            elif y_score[i] < y_score[j]:
                tmp_label = -1
            else:
                tmp_label = 0
            pred_label.append(tmp_label)

    # res
    n_concordant_pairs = sum([1 if i == j else 0 for i, j in zip(golden_label, pred_label)])
    n_discordant_pairs = sum(
        [1 if ((i == 1 and j == -1) or (i == -1 and j == 1)) else 0 for i, j in zip(golden_label, pred_label)])

    N = len(y_score)
    res = (n_concordant_pairs - n_discordant_pairs) / (N * (N - 1) / 2)
    return res


class Example(object):
    def __init__(self, feature, feature2=None, label=None, task=None, lang=None):
        self.feature = feature
        self.label = label
        self.feature2 = feature2
        self.task = task
        self.lang = lang


class CoNLLProcessor(object):
    '''Processor for CoNLL-2003 data set.'''

    def __init__(self):

        self.task_dic = {'ner': 0}


    def get_examples(self, target_language):

        train_examples, dev_examples, dev_target_examples, test_examples = self.read_pkl(target_language)
        return train_examples, dev_examples, dev_target_examples, test_examples

    def get_splits(self, data):
        length = len(data)
        splits = []

        for i in range(240):
            splits.append(str(i))

        random.shuffle(splits)

        train_splits = splits[:int(0.5 * length)]
        dev_splits = splits[int(0.5 * length):int(0.75 * length)]
        test_splits = splits[int(0.75 * length):]
        return train_splits, dev_splits, test_splits

    def read_pkl(self, target_language):
        '''
        read file
        '''
        task_file = "ner_wikiann.pkl"

        with open(task_file, 'rb') as f:
            data = pickle.load(f)

        train_splits1, dev_splits1, test_splits1 = self.get_splits(data)

        train_examples = []
        dev_examples = []
        self.target_language = target_language

        self.lang_dic = {}
        # create training
        # create dev
        #lang_list = 'ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu'
        lang_list = 'tr,ka,es,id,zh,ru'
        target_lang_list = 'qu,cdo,ilo,xmf,mi,mhr,tk,gn'

        lang_list = lang_list.split(',')
        target_lang_list = target_lang_list.split(',')

        for idx, l in enumerate(lang_list):
            self.lang_dic[l] = idx

        for idx, l in enumerate(target_lang_list):
            self.lang_dic[l] = idx + len(lang_list)


        for lang in lang_list:
            tmp_train_examples = self.create_train_examples(data, train_splits1, 'repr_**'.replace('**', lang),
                                                            'dev_**'.replace('**', lang),
                                                            task='ner', lang=lang)
            tmp_dev_examples = self.create_eval_examples(data, dev_splits1, 'repr_**'.replace('**', lang),
                                                         'dev_**'.replace('**', lang), task='ner',
                                                         lang=lang)
            train_examples.extend(tmp_train_examples)
            dev_examples.append(tmp_dev_examples)

        # create development data for target language
        dev_target_examples = self.create_eval_examples(data, dev_splits1, 'repr_**'.replace('**', target_language),
                                                        'dev_**'.replace('**', target_language),
                                                        task='ner', lang=target_language)
        # create test
        test_examples = self.create_eval_examples(data, test_splits1, 'repr_**'.replace('**', target_language),
                                                  'test_**'.replace('**', target_language),
                                                  task='ner', lang=target_language)

        self.run_dev_baseline(data, dev_splits1, 'dev_**'.replace('**', target_language))
        self.run_test_baseline(data, test_splits1, 'test_**'.replace('**', target_language))


        return train_examples, dev_examples, dev_target_examples, test_examples

    def run_test_baseline(self, data, splits, label_f1):

        target_f1 = []
        for i in range(len(splits)):
            model_one = data[splits[i]]
            target_f1.append(model_one[label_f1])

        sorted_f1 = np.sort(target_f1)[::-1]
        self.sorted_test_f1 = sorted_f1

    def run_dev_baseline(self, data, splits, label_f1):

        target_f1 = []
        for i in range(len(splits)):
            model_one = data[splits[i]]
            target_f1.append(model_one[label_f1])

        sorted_f1 = np.sort(target_f1)[::-1]
        self.sorted_dev_f1 = sorted_f1

    def create_train_examples(self, data, splits, feature, label_f1, task, lang):

        #feature_target = 'qa_dev_' + self.target_language + '_repr'
        #feature_en = 'qa_dev_en_repr'
        feature_pivot = 'qa_dev_' + lang + '_repr'

        examples = []
        for i in range(len(splits) - 1):
            for j in range(i + 1, len(splits)):

                model_one = data[splits[i]]
                model_two = data[splits[j]]

                label_f1 = label_f1

                f1_1 = model_one[label_f1]
                f1_2 = model_two[label_f1]

                if f1_1 > f1_2:
                    label = 1  # 1
                elif f1_1 < f1_2:
                    label = 0  # 0
                else:
                    label = 0.5  # 0.5

                # use monolingual feature
                repr_1 = model_one[feature]
                repr_2 = model_two[feature]

                examples.append(Example(feature=repr_1, feature2=repr_2, label=label, lang=self.lang_dic[lang]))

        return examples

    def create_eval_examples(self, data, splits, feature, label, task, lang):
        examples = []


        for i in range(len(splits)):
            model_one = data[splits[i]]
            label = label
            f1 = model_one[label]


            # Monolingual feature
            repr_1 = model_one[feature]

            examples.append(Example(feature=repr_1, label=f1, task=self.task_dic[task], lang=self.lang_dic[lang]))

        return examples


def create_dataloader(batchsize=32, target_language=None):
    proc = CoNLLProcessor()

    train_examples, dev_examples, dev_target_examples, test_examples = proc.get_examples(target_language)

    all_feat_1 = torch.tensor([f.feature for f in train_examples], dtype=torch.float32)
    all_feat_2 = torch.tensor([f.feature2 for f in train_examples], dtype=torch.float32)
    all_label_ids = torch.tensor([f.label for f in train_examples], dtype=torch.float32)

    all_lang_ids = torch.tensor([f.lang for f in train_examples], dtype=torch.long)

    train_dataset = TensorDataset(all_feat_1, all_feat_2, all_label_ids, all_lang_ids)
    data_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=data_sampler, batch_size=batchsize)

    # dev
    dev_dataloaders = []
    for dev_example in dev_examples:
        dev_all_feat_1 = torch.tensor([f.feature for f in dev_example], dtype=torch.float32)
        dev_all_label_ids = torch.tensor([f.label for f in dev_example], dtype=torch.float32)

        dev_all_lang_ids = torch.tensor([f.lang for f in dev_example], dtype=torch.long)

        dev_dataset = TensorDataset(dev_all_feat_1, dev_all_label_ids, dev_all_lang_ids)
        dev_data_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_data_sampler, batch_size=batchsize)
        dev_dataloaders.append(dev_dataloader)

    dev_target_all_feat_1 = torch.tensor([f.feature for f in dev_target_examples], dtype=torch.float32)
    dev_target_all_label_ids = torch.tensor([f.label for f in dev_target_examples], dtype=torch.float32)
    dev_target_all_lang_ids = torch.tensor([f.lang for f in dev_target_examples], dtype=torch.long)

    dev_target_dataset = TensorDataset(dev_target_all_feat_1, dev_target_all_label_ids, dev_target_all_lang_ids)
    dev_target_data_sampler = SequentialSampler(dev_target_dataset)
    dev_target_dataloader = DataLoader(dev_target_dataset, sampler=dev_target_data_sampler, batch_size=batchsize)

    test_all_feat_1 = torch.tensor([f.feature for f in test_examples], dtype=torch.float32)
    test_all_label_ids = torch.tensor([f.label for f in test_examples], dtype=torch.float32)
    test_all_lang_ids = torch.tensor([f.lang for f in test_examples], dtype=torch.long)

    test_dataset = TensorDataset(test_all_feat_1, test_all_label_ids, test_all_lang_ids)
    test_data_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_data_sampler, batch_size=batchsize)

    return train_dataloader, dev_dataloaders, dev_target_dataloader, test_dataloader, proc


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, activation='relu'):
        super(Model, self).__init__()

        act_dict = {'relu': nn.ReLU(),
                    'gelu': nn.GELU()}
        self.act = act_dict[activation]

        self.fc1 = nn.Linear(768, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, repr):
        # (786) --> (hidden_size)
        # (512) --> (hidden_size)
        bz, _ = repr.size()
        out = self.fc1(repr)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        res = self.fc3(out)
        return res

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device


class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size, activation='gelu'):
        super(RankNet, self).__init__()
        # load pretrained embedding
        self.f = Model(input_size, hidden_size, activation)
        self.loss_func = torch.nn.BCELoss()

    def forward(self, repr, repr2, label, task=None, lang=None):
        batchsize, _ = repr.size()
        si = self.f(repr)
        sj = self.f(repr2)

        oij = torch.sigmoid(si - sj)
        label = label.unsqueeze(-1)
        #print(oij)
        #print(label)
        loss = self.loss_func(oij, label)
        return loss.mean()

    def evaluate(self, repr):
        out = self.f(repr)
        return out

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device


def evaluate(model, eval_dataloader):
    model.eval()
    device = model.get_device()

    y_pred = []
    y_true = []
    final_ndcg_result = []
    for repr, label, lang in eval_dataloader:
        y_pred_tmp = []
        y_true_tmp = []
        repr = repr.to(device)

        with torch.no_grad():
            score = model.evaluate(repr)

        score = score.detach().cpu().numpy()

        for s in score:
            y_pred_tmp.append(s[0])

        label = label.detach().cpu().numpy()
        for t in label:
            y_true_tmp.append(t)

        ndcg_result = ndcg_score(y_true_tmp, y_pred_tmp)
        final_ndcg_result.append(ndcg_result)

        y_pred.extend(y_pred_tmp)
        y_true.extend(y_true_tmp)

    idx = np.argmax(y_pred)
    #print(idx)
    #print(y_true)
    final_model_pred = y_true[idx]
    ndcg_final_score = sum(final_ndcg_result) / len(final_ndcg_result)

    # kendall coefficient
    kendal_score = kendall_rank_correlation(y_true, y_pred)

    sort_y = np.sort(y_true)[::-1]
    # print(sort_y)
    rank = list(sort_y).index(final_model_pred)
    return final_model_pred, ndcg_final_score, kendal_score, rank


def train(model, train_dataloader=None, dev_dataloaders=None,
          optimizer=None, max_epochs=None,
          save_ckpt=None, save_config=None):
    device = model.get_device()

    best_acc = -1

    for epoch in range(max_epochs):
        train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            repr, repr2, label, lang = batch

            loss = model(repr=repr, repr2=repr2, label=label, lang=lang)
            #print(loss)
            #print('Epoch: %d, step: %d, training loss: %.5f' % (epoch, step, loss.item()))
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # model_to_save = model.module if hasattr(model, 'module') else model
        # output_model_file = save_ckpt
        # torch.save(model_to_save.state_dict(), output_model_file)

        avg_train_loss = train_loss / len(train_dataloader)

        print("Epoch: %d, average training loss: %.5f" % (epoch, avg_train_loss))
        final_kendall_score = 0
        for dev_dataloader in dev_dataloaders:
            final_model_pred, acc_cur, kendall_score, _ = evaluate(model, dev_dataloader)
            final_kendall_score += kendall_score
        final_kendall_score /= len(dev_dataloaders)
        print("Epoch: %d, kendall: %.5f" % (epoch, final_kendall_score))

        # Save best model
        if final_kendall_score > best_acc:
            best_acc = final_kendall_score
            # save best chpt
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = save_ckpt
            torch.save(model_to_save.state_dict(), output_model_file)

    return model
