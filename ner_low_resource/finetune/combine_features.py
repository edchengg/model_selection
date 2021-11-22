import numpy as np
import pickle

dir = '/srv/share4/ychen3411/project00_model/'
L = 'ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu,qu,cdo,ilo,xmf,mi,mhr,tk,gn'
L = L.split(',')

save = {}

def read_results(file):

    f = open(file, 'r').readlines()
    res = {}

    langs = []
    f1s = []
    for line in f:
        line = line.strip('\n')
        if 'language' in line:
            lang = line.split('=')[-1]
            langs.append(lang)
        if 'f1' in line:
            f1 = float(line.split('=')[-1])
            f1s.append(f1)

    for l, f11 in zip(langs, f1s):
        res[l] = f11

    return res

for idx in range(240):
    save[str(idx)] = {}
    for lang in L:
        tmp = dir + str(idx) + '/repr_{}.npz.npy'.format(lang)
        repr = np.load(tmp)
        cond = np.isnan(repr).any()
        if cond:
            print(idx, lang)
        save[str(idx)]['repr_{}'.format(lang)] = repr

for idx in range(240):
    dev_tmp = dir + str(idx) + '/48_dev_results.txt'
    test_tmp = dir + str(idx) + '/48_test_results.txt'

    dev_res = read_results(dev_tmp)
    test_res = read_results(test_tmp)

    print(dev_res)
    for key, v in dev_res.items():
        save[str(idx)]['dev_' + key] = v

    for key, v in test_res.items():
        save[str(idx)]['test_' + key] = v

with open('/srv/scratch/ychen3411/project00_model_selection/lms/feature.pkl', 'wb') as f:
    pickle.dump(save, f, protocol=pickle.HIGHEST_PROTOCOL)


