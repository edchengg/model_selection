'''
Collect representation and Accuracy
'''
import numpy as np
import pickle

save_dir = 'evaluation/{}/{}.txt'
save_dir_repr = 'evaluation/{}/{}.npy'

dict = {}

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

for idx in range(240):
    print(idx)
    dict[idx] = {}
    for lang in list(lang2full.keys()):

        res = save_dir.format(idx, lang)
        repr = save_dir_repr.format(idx, lang)

        file = open(res, 'r').readlines()
        dev_acc = file[0].split('=')[-1]
        test_acc = file[1].split('=')[-1]

        dict[idx]['{}_dev'.format(lang)] = float(dev_acc)
        dict[idx]['{}_test'.format(lang)] = float(test_acc)

        repr_np = np.load(repr)
        dict[idx]['{}_repr'.format(lang)] = repr_np

with open('pos_feature.pkl', 'wb') as f:
    pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)
