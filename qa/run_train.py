import os

print('Run training for ar, de, es, zh')
for lang in ['ar', 'de', 'es', 'zh']:
    script = 'python main2.py --exp_name qa-{}-lms --target_language {}'.format(lang, lang)

    os.system(script)

print('Run training for hi,vi')
script = 'python main3.py --exp_name qa-vi,hi-lms --target_language vi'
os.system(script)