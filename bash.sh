#!/usr/bin/env bash

# Baseline
#pos
echo "===POS - Table 3==="
cd pos/lms_table3
python baseline.py
python baseline_bg-sv.py
python run_evaluation.py
#qa
echo "===QA - Table 3==="
cd ../../qa
python baseline.py
python run_evaluation.py
#re
echo "===RE - Table 3==="
cd ../re
python baseline.py
python run_evaluation.py
cd..
#arl
echo "===ARL - Table 3==="
cd arl
python baseline.py
python run_evaluation.py
#conll
echo "===NER CoNLL - Table 3==="
cd ..
cd ner_conll
python baseline.py
python run_evaluation.py
#low-resource ner
echo "===NER Low Resource - Table 3==="
cd ..
cd ner_low_resource/lms_table3
python baseline.py
python evaluation.py
#pos-multilingual
echo "===POS Multilingual - Table 4==="
cd ..
cd ..
cd pos/lms_table4
python baseline.py
python evaluation.py
