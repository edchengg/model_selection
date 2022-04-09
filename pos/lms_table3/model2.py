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
    n_discordant_pairs = sum([1 if ((i == 1 and j == -1) or (i == -1 and j == 1)) else 0 for i, j in zip(golden_label, pred_label)])

    N = len(y_score)
    res = (n_concordant_pairs - n_discordant_pairs) / (N*(N-1)/2)
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
    def __init__(self, use_en):

        self.task_dic = {'ner': 0, 'pos': 1}
        self.use_en = use_en

    def get_examples(self, target_task, target_language, feature1, feature2):

        train_examples, dev_examples, dev_target_examples, test_examples = self.read_pkl(target_task, target_language, feature1, feature2)
        return train_examples, dev_examples, dev_target_examples, test_examples

    def get_splits(self, data):
        length = len(data)
        splits = []
        for k, v in data.items():
            splits.append(k)

        random.shuffle(splits)

        train_splits = splits[:int(0.5 * length)]
        dev_splits = splits[int(0.5 * length):int(0.75 * length)]
        test_splits = splits[int(0.75 * length):]
        return train_splits, dev_splits, test_splits

    def read_pkl(self, target_task, target_language, feature1, feature2):
        '''
        read file
        '''
        if target_task == 'conll':
            task_file = ""
        else:
            task_file = "pos18.pkl"

        with open(task_file, 'rb') as f:
            data = pickle.load(f)

        train_splits1, dev_splits1, test_splits1 = self.get_splits(data)
        #print(test_splits1)
        train_examples = []
        dev_examples = []
        self.target_language = target_language

        if target_task == 'conll':
            self.lang_dic = {'de': 0, 'es': 1, 'nl': 2, 'zh': 3, 'en':4}
            # create training
            # create dev
            lang_list = ['de', 'es', 'nl', 'zh']
            lang_list.remove(target_language)
            if self.use_en == 1:
                lang_list.append('en')

            for lang in lang_list:
                tmp_train_examples = self.create_train_examples(data, train_splits1, feature1.replace('**', lang),
                                                                'conll_dev_**_f1'.replace('**', lang),
                                                                task='ner', lang=lang)
                tmp_dev_examples = self.create_eval_examples(data, dev_splits1, feature1.replace('**', lang),
                                                             'conll_dev_**_f1'.replace('**', lang), task='ner',
                                                             lang=lang)
                train_examples.extend(tmp_train_examples)
                dev_examples.append(tmp_dev_examples)

            # create development data for target language
            dev_target_examples = self.create_eval_examples(data, dev_splits1, feature1.replace('**', target_language),
                                                          'conll_dev_**_f1'.replace('**', target_language),
                                                          task='ner', lang=target_language)
            # create test
            test_examples = self.create_eval_examples(data, test_splits1, feature1.replace('**', target_language),
                                                      'conll_test_**_f1'.replace('**', target_language),
                                                      task='ner', lang=target_language)

            self.run_dev_baseline(data, dev_splits1, 'conll_dev_**_f1'.replace('**', target_language))
            self.run_test_baseline(data, test_splits1, 'conll_test_**_f1'.replace('**', target_language))
        else:
            self.lang_dic = {'ar': 0, 'de': 1, 'es': 2, 'nl': 3, 'zh': 4,
                             'bg': 5, 'da': 6, 'fa': 7, 'hu': 8, 'it': 9, 'pt': 10,
                             'ro': 11, 'sk': 12, 'sl': 13, 'sv': 14}
            # create training
            # create dev
            lang_list = ['ar', 'de', 'es', 'nl', 'zh']

            if self.use_en == 1:
                lang_list.append('en')

            for lang in lang_list:
                tmp_train_examples = self.create_train_examples(data, train_splits1, feature2.replace('**', lang),
                                                                'ud_dev_**_acc'.replace('**', lang), task='pos',
                                                                lang=lang)
                tmp_dev_examples = self.create_eval_examples(data, dev_splits1, feature2.replace('**', lang),
                                                             'ud_dev_**_acc'.replace('**', lang), task='pos', lang=lang)
                train_examples.extend(tmp_train_examples)
                dev_examples.append(tmp_dev_examples)

            # create test
            dev_target_examples = self.create_eval_examples(data, dev_splits1, feature2.replace('**', target_language),
                                                      'ud_dev_**_acc'.replace('**', target_language), task='pos',
                                                      lang=target_language)
            # create test
            test_examples = self.create_eval_examples(data, test_splits1, feature2.replace('**', target_language),
                                                      'ud_test_**_acc'.replace('**', target_language), task='pos',
                                                      lang=target_language)

            self.run_dev_baseline(data, dev_splits1, 'ud_dev_**_acc'.replace('**', target_language))
            self.run_test_baseline(data, test_splits1, 'ud_test_**_acc'.replace('**', target_language))

        # print('Train data size: ', len(train_examples))
        # print('Dev data size: ', len(dev_examples) * 60)
        # print('Test data size: ', len(test_examples))
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
        if task == 'ner':
            feature_target = 'conll_' + self.target_language + '_dev_repr'
            feature_en = 'conll_en_dev_repr'
            feature_pivot = 'conll_' + lang + '_dev_repr'
            feature_paral = 'conll_en2' + lang + '_repr'
        elif task == 'pos':
            feature_target = 'ud_' + self.target_language + '_dev_repr'
            feature_en = 'ud_en_dev_repr'
            feature_pivot = 'ud_' + lang + '_dev_repr'
            feature_paral = 'ud_en2' + lang + '_repr'

        examples = []
        for i in range(len(splits) - 1):
            for j in range(i + 1, len(splits)):

                model_one = data[splits[i]]
                model_two = data[splits[j]]

                if lang != 'en':
                    label_f1 = label_f1
                else:
                    label_f1 = 'conll_dev_f1'

                f1_1 = model_one[label_f1]
                f1_2 = model_two[label_f1]

                if f1_1 > f1_2:
                    label = 1
                elif f1_1 < f1_2:
                    label = -1
                else:
                    label = 0

                # Use parallel feature
                if '2' in feature:
                    repr_1_1 = model_one[feature]['en']['mean']
                    repr_1_2 = model_one[feature][lang]['mean']
                    repr_1 = repr_1_1 + repr_1_2


                    repr_2_2 = model_two[feature][lang]['mean']
                    repr_2_1 = model_two[feature]['en']['mean']
                    repr_2 = repr_2_1 + repr_2_2
                elif feature == 'all_mono':
                    # concat all feature
                    repr_1_target = model_one[feature_target]
                    repr_1_en = model_one[feature_en]
                    repr_1_pivot = model_one[feature_pivot]

                    repr_1 = np.concatenate((repr_1_target, repr_1_en, repr_1_pivot))

                    repr_2_target = model_two[feature_target]
                    repr_2_en = model_two[feature_en]
                    repr_2_pivot = model_two[feature_pivot]

                    repr_2 = np.concatenate((repr_2_target, repr_2_en, repr_2_pivot))

                elif feature == 'all':
                    # concat all feature
                    repr_1_target = model_one[feature_target]
                    repr_1_en = model_one[feature_en]
                    repr_1_pivot = model_one[feature_pivot]

                    repr_1_paral_en = model_one[feature_paral]['en']['mean']
                    repr_1_paral_p = model_one[feature_paral][lang]['mean']
                    repr_1_paral = repr_1_paral_en + repr_1_paral_p

                    repr_1 = np.concatenate((repr_1_target, repr_1_en, repr_1_pivot, repr_1_paral))

                    repr_2_target = model_two[feature_target]
                    repr_2_en = model_two[feature_en]
                    repr_2_pivot = model_two[feature_pivot]

                    repr_2_paral_en = model_one[feature_paral]['en']['mean']
                    repr_2_paral_p = model_one[feature_paral][lang]['mean']
                    repr_2_paral = repr_2_paral_en + repr_2_paral_p
                    repr_2 = np.concatenate((repr_2_target, repr_2_en, repr_2_pivot, repr_2_paral))

                else:
                    # use monolingual feature
                    repr_1 = model_one[feature]
                    repr_2 = model_two[feature]

                examples.append(Example(feature=repr_1, feature2=repr_2, label=label, lang=self.lang_dic[lang]))

        return examples

    def create_eval_examples(self, data, splits, feature, label, task, lang):
        examples = []
        if task == 'ner':
            feature_target = 'conll_' + self.target_language + '_dev_repr'
            feature_en = 'conll_en_dev_repr'
            feature_pivot = 'conll_' + lang + '_dev_repr'
            feature_paral = 'conll_en2' + lang + '_repr'
        elif task == 'pos':
            feature_target = 'ud_' + self.target_language + '_dev_repr'
            feature_en = 'ud_en_dev_repr'
            feature_pivot = 'ud_' + lang + '_dev_repr'
            feature_paral = 'ud_en2' + lang + '_repr'

        for i in range(len(splits)):
            model_one = data[splits[i]]
            if lang != 'en':
                label = label
            else:
                label = 'conll_dev_f1'
            f1 = model_one[label]

            # Parallel feature
            if '2' in feature:
                repr_1_1 = model_one[feature]['en']['mean']
                repr_1_2 = model_one[feature][lang]['mean']
                repr_1 = repr_1_1 + repr_1_2
            elif feature == 'all_mono':
                # concat all feature
                repr_1_target = model_one[feature_target]
                repr_1_en = model_one[feature_en]
                repr_1_pivot = model_one[feature_pivot]

                repr_1 = np.concatenate((repr_1_target, repr_1_en, repr_1_pivot))

            elif feature == 'all':
                # concat all feature
                repr_1_target = model_one[feature_target]
                repr_1_en = model_one[feature_en]
                repr_1_pivot = model_one[feature_pivot]

                repr_1_paral_en = model_one[feature_paral]['en']['mean']
                repr_1_paral_p = model_one[feature_paral][lang]['mean']
                repr_1_paral = repr_1_paral_en + repr_1_paral_p

                repr_1 = np.concatenate((repr_1_target, repr_1_en, repr_1_pivot, repr_1_paral))

            else:
                # Monolingual feature
                repr_1 = model_one[feature]

            examples.append(Example(feature=repr_1, label=f1, task=self.task_dic[task], lang=self.lang_dic[lang]))

        return examples


def create_dataloader(batchsize=32, target_task=None, target_language=None, feature1=None, feature2=None, use_en=0):

    proc = CoNLLProcessor(use_en=use_en)


    train_examples, dev_examples, dev_target_examples, test_examples = proc.get_examples(target_task, target_language, feature1, feature2)

    all_feat_1 = torch.tensor([f.feature for f in train_examples], dtype=torch.float32)
    all_feat_2 = torch.tensor([f.feature2 for f in train_examples], dtype=torch.float32)
    all_label_ids = torch.tensor([f.label for f in train_examples], dtype=torch.float32)

    all_lang_ids = torch.tensor([f.lang for f in train_examples], dtype=torch.long)

    train_dataset = TensorDataset(all_feat_1, all_feat_2, all_label_ids,  all_lang_ids)
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

    test_dataset = TensorDataset(test_all_feat_1, test_all_label_ids,  test_all_lang_ids)
    test_data_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_data_sampler, batch_size=batchsize)

    return train_dataloader, dev_dataloaders, dev_target_dataloader, test_dataloader, proc


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, emb_dim, activation='relu', weighted_sum=0):
        super(Model, self).__init__()

        act_dict = {'relu': nn.ReLU(),
         'gelu': nn.GELU()}
        self.act = act_dict[activation]

        # all features together
        if weighted_sum == 1:
            self.n_feature = input_size // 768
            # weighted sum feature
            self.softmax_weight = nn.Parameter(torch.empty(768, self.n_feature))
        self.weighted_sum = weighted_sum

        self.fc1 = nn.Linear(768, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc3 = nn.Linear(emb_dim, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bilinear = nn.Parameter(torch.empty((hidden_size, hidden_size)))

        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.bilinear)
        if self.weighted_sum == 1:
            nn.init.xavier_normal_(self.softmax_weight)

    def forward(self, repr, emb):
        # (786) --> (hidden_size)
        # (512) --> (hidden_size)
        bz, _ = repr.size()
        if self.weighted_sum == 1:
            weight = F.softmax(self.softmax_weight, dim=-1)
            repr = repr.view(bz, self.n_feature, -1).transpose(1, 2)
            repr = weight * repr
            repr = torch.sum(repr, dim=-1)

        out = self.fc1(repr)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)

        emb = self.fc3(emb)
        emb = self.act(emb)
        emb = self.fc4(emb)
        emb = self.act(emb).unsqueeze(2)

        res =  out.matmul(self.bilinear).unsqueeze(1)
        res = res.matmul(emb).squeeze(2)
        res = torch.sigmoid(res)
        return res

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device

class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, activation='gelu', emb_dim=768, weighted_sum=0):
        super(RankNet, self).__init__()
        self.lang_embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        # load pretrained embedding
        self.f = Model(input_size, hidden_size, emb_dim, activation, weighted_sum)
        self.loss_func = torch.nn.BCELoss()
    def forward(self, repr, repr2, label, task=None, lang=None):
        batchsize, _ = repr.size()

        lang_emb = self.lang_embedding(lang)
        #repr = torch.cat((repr, lang_emb), dim=-1)
        #repr2 = torch.cat((repr2, lang_emb), dim=-1)
        si = self.f(repr, lang_emb)
        sj = self.f(repr2, lang_emb)

        oij = torch.sigmoid(si - sj)
        label = label.unsqueeze(-1)
        # P_hat = 0.5 * ( 1 + Sij) --> label trans to 0,0.5, 1
        #P_hat = 0.5 * (torch.ones(size=(batchsize, 1), device=self.device, dtype=torch.float32) + label)
        #print(P_hat.size())
        # loss from paper:
        # Cij = - P_hat * oij + log (1 + e^ oij)
        #loss = - P_hat * oij + torch.log(1 + torch.exp(oij))
        loss = self.loss_func(oij, label)
        return loss.mean()

    def evaluate(self, repr, lang=None):

        lang_emb = self.lang_embedding(lang)
        out = self.f(repr, lang_emb)
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

        lang = lang.to(device)
        with torch.no_grad():
            score = model.evaluate(repr, lang)

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

    final_model_pred = y_true[idx]
    ndcg_final_score = sum(final_ndcg_result)/len(final_ndcg_result)

    #kendall coefficient
    kendal_score = kendall_rank_correlation(y_true, y_pred)
    return final_model_pred, ndcg_final_score, kendal_score


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

            print('Epoch: %d, step: %d, training loss: %.5f' % (epoch, step, loss.item()))
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = save_ckpt
        torch.save(model_to_save.state_dict(), output_model_file)

        avg_train_loss = train_loss / len(train_dataloader)

        print("Epoch: %d, average training loss: %.5f" % (epoch, avg_train_loss))
        final_kendall_score = 0
        for dev_dataloader in dev_dataloaders:
            final_model_pred, acc_cur, kendall_score = evaluate(model, dev_dataloader)
            final_kendall_score += kendall_score
        final_kendall_score/=len(dev_dataloaders)
        print("Epoch: %d, kendall: %.5f" % (epoch, final_kendall_score))

        # Save best model
        if final_kendall_score > best_acc:
            best_acc = final_kendall_score
            # save best chpt
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = 'best_' + save_ckpt
            torch.save(model_to_save.state_dict(), output_model_file)

    return model
