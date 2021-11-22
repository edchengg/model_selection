import os
import argparse
import random

parser = argparse.ArgumentParser()
## Required parameters

parser.add_argument("--sh_dir", default='run_0809', type=str,
                    help="output file name")
args = parser.parse_args()




if not os.path.exists(args.sh_dir):
    os.mkdir(args.sh_dir)

tmp_file = open('{}/p1.sh'.format(args.sh_dir), 'w')

tmp_file.write('#!/bin/bash -l\n')
tmp_file.write('#SBATCH -J pos\n')
tmp_file.write('#SBATCH --output=log/0809-%a.out\n')
tmp_file.write('#SBATCH --error=log/0809-%a.err\n')
tmp_file.write('#SBATCH --gres=gpu:1\n')
tmp_file.write('#SBATCH -c 6\n')
tmp_file.write('#SBATCH --partition=overcap\n')
tmp_file.write('#SBATCH --account=overcap\n')
tmp_file.write('#SBATCH --constraint="titan_x|2080_ti|rtx_6000|a40"\n')
tmp_file.write('#SBATCH -a 0-240\n')
tmp_file.write('#SBATCH --requeue\n')


tmp_file.write("sid=$SLURM_ARRAY_TASK_ID\n")
tmp_file.write("echo SID:$sid\n")
tmp_file.write("LR=$(sed '1q;d' {}/config_$sid.txt)\n".format(args.sh_dir))
tmp_file.write("BZ=$(sed '2q;d' {}/config_$sid.txt)\n".format(args.sh_dir))
tmp_file.write("EPOCH=$(sed '3q;d' {}/config_$sid.txt)\n".format(args.sh_dir))
tmp_file.write("SEED=$(sed '4q;d' {}/config_$sid.txt)\n".format(args.sh_dir))



script = 'python main_multilingual.py --exp_name $SEED --learning_rate $LR --batchsize $BZ --max_epoch $EPOCH'
tmp_file.write(script + '\n')
tmp_file.close()

job_id = 0

for _ in range(20):
    for lr in [2e-5, 3e-5, 5e-5, 7e-5]:
        for epoch in [3, 5, 7]:

            config_tmp = open('{}/config_{}.txt'.format(args.sh_dir, job_id), 'w')
            config_tmp.write('{}\n'.format(lr))
            config_tmp.write('{}\n'.format(32))
            config_tmp.write('{}\n'.format(epoch))
            config_tmp.write('{}\n'.format(job_id))
            config_tmp.close()
            job_id += 1