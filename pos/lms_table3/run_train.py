import os

print('Run training for ar, de, es, nl, zh')
for lang in ['ar', 'de', 'es', 'nl', 'zh']:
    script = 'python main.py --exp_name pos-{}-lms --target_language {}'.format(lang, lang)

    os.system(script)

print('Run training for bg-sv')
script = 'python main2.py --exp_name pos-bg-sv-lms --target_language bg'
os.system(script)