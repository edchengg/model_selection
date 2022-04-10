import os
import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--exp_name", default='tb3-lowresource', type=str,
                    help="Checkpoint and config save prefix")
args = parser.parse_args()

print('Loading ckpt and reproduce results for Table 3 - Low resource...')

if os.path.exists('{}.ckpt'.format(args.exp_name)):
    pass
else:
    print('ckpt not exists')
    NotImplementedError()

for lang in ['cdo', 'gn', 'ilo', 'mhr', 'mi', 'tk', 'qu', 'xmf']:
    os.system('python main_load_evaluate.py --target_language {} --exp_name {}'.format(lang, args.exp_name))