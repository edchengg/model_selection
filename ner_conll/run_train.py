import os

print('Run training for de, es, nl, zh')

script = 'python main.py --exp_name pos-de-lms --target_language de --hidden_size 512 --max_epoch 5 --learning_rate 1e-6 --batchsize 32'
os.system(script)
script = 'python main.py --exp_name pos-es-lms --target_language es --hidden_size 512 --max_epoch 5 --learning_rate 1e-6 --batchsize 32'
os.system(script)
script = 'python main.py --exp_name pos-nl-lms --target_language nl --hidden_size 512 --max_epoch 5 --learning_rate 1e-5 --batchsize 32'
os.system(script)
script = 'python main.py --exp_name pos-zh-lms --target_language zh --hidden_size 512 --max_epoch 5 --learning_rate 1e-6 --batchsize 32'
os.system(script)