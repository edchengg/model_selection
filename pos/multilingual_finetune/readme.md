# POS - Table 4

Section 6.2 (Evaluation on multilingual fine-tuned models)

This repo is for multilingual fine-tuning on POS and extracting [CLS] representation from all checkpoints.

## Training
``
python main_multilingual.py
``

## Extracting feature + evaluation
``
python main_evalute.py
``

## Collect feature and data into a pickle file
``
python collect_feature_and_stats.py
``

The final saved ckpt file: pos_feature.pkl is then used in the lms_table4/ folder.