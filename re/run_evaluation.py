
import os
import argparse



print('Loading ckpt and reproduce results for Table 3 RE...')

for lang in ['ar', 'zh']:
    if os.path.exists('re-{}-lms.ckpt'.format(lang)):
        pass
    else:
        print('ckpt not exists')
        NotImplementedError()



for lang in ['ar', 'zh']:
    os.system('python main_load_evaluate.py --target_language {} --exp_name {}'.format(lang, 're-{}-lms'.format(lang)))