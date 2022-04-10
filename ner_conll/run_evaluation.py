
import os

print('Loading ckpt and reproduce results for Table 3 NER CoNLL...')

for lang in ['de', 'es', 'nl', 'zh']:
    if os.path.exists('{}.ckpt'.format('ner-{}-lms'.format(lang))):
        pass
    else:
        print('ckpt not exists')
        NotImplementedError()


for lang in ['de', 'es', 'nl', 'zh']:
    os.system('python main_load_evaluate.py --target_language {} --exp_name {}'.format(lang, 'ner-{}-lms'.format(lang)))
