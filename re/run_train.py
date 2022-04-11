import os

os.system("python main.py --exp_name re-ar-lms --max_epoch 5 --target_language ar --learning_rate 1e-6 --batchsize 16")
os.system("python main.py --exp_name re-zh-lms --max_epoch 3 --target_language zh --learning_rate 1e-5 --batchsize 32")