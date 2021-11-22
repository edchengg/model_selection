# POS - Table 4

This repo reproduces the results in Table 4 - Section 6.2 (Evaluation on multilingual fine-tuned models)

## Baseline
``
python baseline.py
``

## Training
``
python main_train_lms.py --exp_name tb4-ft
``

## Evaluation
``
python evaluation.py --exp_name tb4-ft
``

## Checkpoint
ft-tb4.ckpt