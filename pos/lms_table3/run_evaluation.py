
import os
import argparse





print('Loading ckpt and reproduce results for Table 3 POS...')

for lang in ['ar', 'de', 'es', 'zh', 'bg-sv']:
    if os.path.exists('{}.ckpt'.format('pos-{}-lms'.format(lang))):
        pass
    else:
        print('ckpt not exists')
        NotImplementedError()


for lang in ['ar', 'de', 'es', 'nl', 'zh']:
    os.system('python main_load_evaluate.py --target_language {} --exp_name {}'.format(lang, 'pos-{}-lms'.format(lang)))

for lang in ['bg', 'da', 'fa', 'hu', 'it', 'pt', 'ro', 'sk', 'sl', 'sv']:
    os.system('python main_load_evaluate2.py --target_language {} --exp_name {}'.format(lang, 'pos-bg-sv-lms'))