#!/usr/bin/env bash

# Baseline
#pos
cd pos/lms_table3
python baseline.py
python baseline_bg-sv.py
python run_evaluation.py
#qa
cd ../../qa
python baseline.py
python run_evaluation.py
#conll
cd ..
cd ..
cd ner_conll
python baseline.py
python run_evaluation.py
#low-resource ner
cd ..
cd ner_low_resource/lms_table3
python baseline.py
python evaluation.py
#pos-multilingual
cd ..
cd ..
cd pos/lms_table4
python baseline.py
python evaluation.py
