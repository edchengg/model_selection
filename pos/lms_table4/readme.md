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

=============================================\\
Loading ckpt and reproduce results for Table 4...\\
Loading ckpt: tb4-ft.ckpt...\\
ckpt exists\\
Target lang: bg, TEST ACC: 0.90759\\
Target lang: da, TEST ACC: 0.89293\\
Target lang: fa, TEST ACC: 0.81057\\
Target lang: hu, TEST ACC: 0.83991\\
Target lang: it, TEST ACC: 0.94940\\
Target lang: pt, TEST ACC: 0.91072\\
Target lang: ro, TEST ACC: 0.88787\\
Target lang: sk, TEST ACC: 0.88617\\
Target lang: sl, TEST ACC: 0.86752\\
Target lang: sv, TEST ACC: 0.92153\\

=============================================


## Checkpoint
ft-tb4.ckpt