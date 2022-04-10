import os
import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--exp_name", default='tb4-ft', type=str,
                    help="Checkpoint and config save prefix")
args = parser.parse_args()

print('Loading ckpt and reproduce results for Table 4...')

if os.path.exists('{}.ckpt'.format(args.exp_name)):
    pass
else:
    print('ckpt not exists')
    NotImplementedError()

for lang in ['bg', 'da', 'fa', 'hu', 'it', 'pt', 'ro', 'sk', 'sl', 'sv']:
    os.system('python main_load_evaluate.py --target_language {} --exp_name {}'.format(lang, args.exp_name))