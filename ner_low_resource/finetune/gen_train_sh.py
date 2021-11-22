

seed = 0

for sh in range(10):
    tmp_file = open('train_bash/run_{}.sh'.format(sh), 'w')

    tmp_file.write('#!/bin/sh -l\n')
    tmp_file.write('#SBATCH --job-name=ft-{}\n'.format(sh))
    tmp_file.write('#SBATCH --output=ft-{}-log\n'.format(sh))
    tmp_file.write('#SBATCH --error=ft-{}-err\n'.format(sh))
    tmp_file.write('#SBATCH --gres=gpu:1\n')
    tmp_file.write('#SBATCH --exclude=rosie\n')
    tmp_file.write('#SBATCH -c 6\n')
    tmp_file.write('#SBATCH --partition=short\n')
    tmp_file.write('#SBATCH --constraint=2080_ti\n')
    # tmp_file.write('#SBATCH --signal=USR1@300\n')
    # tmp_file.write('#squeeSBATCH --requeue\n')
    # tmp_file.write('#SBATCH --account=overcap\n')
    tmp_file.write('source ~/.bashrc\nsource activate pytorch_p37\n')
    tmp_file.write('REPO=$PWD\n')
    tmp_file.write('LANG=ar,de,es,nl,zh,qu,cdo,ilo,xmf,mi,mhr,tk,gn\n')


    for _ in range(2):
        for lr in [2e-5, 3e-5, 4e-5, 5e-5]:
            for num_epoch in [6, 8, 10]:
                output_dir = '/srv/share4/ychen3411/project00_model/{}'.format(seed)
                tmp_file.write('python $REPO/run_tag.py \
--model_type bert --model_name_or_path bert-base-multilingual-cased \
--output_dir {} --max_seq_length 128 --num_train_epochs {} \
--gradient_accumulation_steps 1 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --seed {} --learning_rate {} \
--do_train  --do_predict_dev --predict_langs $LANG --train_langs en \
--eval_all_checkpoints --eval_patience -1 --overwrite_output_dir \
--save_only_best_checkpoint --do_predict'.format(output_dir, num_epoch, seed, lr))
                tmp_file.write('\n\n')
                seed += 1
    tmp_file.close()