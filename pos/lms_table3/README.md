# POS - Table 3

This repo reproduces the results in Table 3 - POS

## Baseline
``
python baseline.py
python baseline_bg-sv.py
``

```
====================
lang --- en-dev --- pivot-dev --- all-target
====================
ar--49.7--50.3--52.7
====================
de--89.3--88.7--90.0
====================
es--84.8--84.9--85.1
====================
nl--75.7--75.9--76.0
====================
zh--66.9--68.1--68.8
====================
bg--87.1--87.0--87.9
====================
da--88.6--88.8--89.2
====================
fa--71.6--71.6--73.8
====================
hu--82.5--82.1--83.1
====================
it--84.5--84.9--85.8
====================
pt--81.8--81.8--82.2
====================
ro--83.8--84.2--85.4
====================
sk--83.7--83.6--84.8
====================
sl--84.5--83.5--85.5
====================
sv--91.4--91.8--91.8

```

## Training
``
python run_train.py
``

## Evaluation
``
python run_evaluation.py
``

```
Loading ckpt and reproduce results for Table 3 POS...
Target lang: ar, TEST ACC: 51.6
Target lang: de, TEST ACC: 89.7
Target lang: es, TEST ACC: 85.5
Target lang: nl, TEST ACC: 75.8
Target lang: zh, TEST ACC: 68.3
Target lang: bg, TEST ACC: 87.9
Target lang: da, TEST ACC: 88.9
Target lang: fa, TEST ACC: 73.6
Target lang: hu, TEST ACC: 83.3
Target lang: it, TEST ACC: 84.8
Target lang: pt, TEST ACC: 82.2
Target lang: ro, TEST ACC: 84.7
Target lang: sk, TEST ACC: 84.2
Target lang: sl, TEST ACC: 85.2
Target lang: sv, TEST ACC: 91.9

```
![ScreenShot](table3-pos.png)