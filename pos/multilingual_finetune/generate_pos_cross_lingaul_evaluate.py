import os
import argparse
import random

parser = argparse.ArgumentParser()
## Required parameters

parser.add_argument("--sh_dir", default='run_0810', type=str,
                    help="output file name")
args = parser.parse_args()




if not os.path.exists(args.sh_dir):
    os.mkdir(args.sh_dir)

tmp_file = open('{}/p1.sh'.format(args.sh_dir), 'w')

tmp_file.write('#!/bin/bash -l\n')
tmp_file.write('#SBATCH -J pos\n')
tmp_file.write('#SBATCH --output=log/0810-%a.out\n')
tmp_file.write('#SBATCH --error=log/0810-%a.err\n')
tmp_file.write('#SBATCH --gres=gpu:1\n')
tmp_file.write('#SBATCH -c 6\n')
tmp_file.write('#SBATCH --partition=overcap\n')
tmp_file.write('#SBATCH --account=overcap\n')
tmp_file.write('#SBATCH --constraint="titan_x|2080_ti|rtx_6000|a40"\n')
tmp_file.write('#SBATCH -a 0-240\n')
tmp_file.write('#SBATCH --requeue\n')


tmp_file.write("sid=$SLURM_ARRAY_TASK_ID\n")
tmp_file.write("echo SID:$sid\n")
script = 'python main_evaluate.py --exp_name $sid'
tmp_file.write(script + '\n')
tmp_file.close()
