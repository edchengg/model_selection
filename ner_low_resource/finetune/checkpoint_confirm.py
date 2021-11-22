import os

dir = '/srv/share4/ychen3411/project00_model/'
for idx in range(240):
    tmp = dir + str(idx) + '/pytorch_model.bin'
    tmp2 = dir + str(idx) + '/48_test_results.txt'
    tmp3 = dir + str(idx) + '/48_dev_results.txt'

    cond =  os.path.exists(tmp) and os.path.exists(tmp2) and os.path.exists(tmp3)
    if not cond:
        print(idx)

