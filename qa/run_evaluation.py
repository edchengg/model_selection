
import os
import argparse





print('Loading ckpt and reproduce results for Table 3 QA...')

for lang in ['ar', 'de', 'es', 'zh', 'vi,hi']:
    if os.path.exists('{}.ckpt'.format('qa-{}-lms'.format(lang))):
        pass
    else:
        print('ckpt not exists')
        NotImplementedError()



for lang in ['ar', 'de', 'es', 'zh']:
    os.system('python main_load_evaluate.py --target_language {} --exp_name {}'.format(lang, 'qa-{}-lms'.format(lang)))

for lang in ['hi', 'vi']:
    os.system('python main_load_evaluate3.py --target_language {} --exp_name {}'.format(lang, 'qa-vi,hi-lms'))