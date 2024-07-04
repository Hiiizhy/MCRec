# MCRec

This code contains Pytorch implementation of MCRec:

> Multi-scale Context-aware User Interest Learning for Behavior Pattern Modeling. Zhiying Deng, Jianjun Li, Li Zou, Wei Liu, Si Shi, Qian Chen, Juan Zhao, Guohui Li. The 29th International Conference on Database Systems for Advanced Applications (DASFAA 2024).

MCRec utilizes stacked tensor mapping to learn latent space representations for basket sequences. It employs vertical, stacked dilated, and horizontal convolutions to capture multi-scale context-aware user interest, capturing both short-term and long-term dependencies in sequential patterns with varying item distances. Additionally, MCRec integrates a user-interest adaptive fusion mechanism, combining historical user representations with interaction frequency preferences for accurate predictions.

## Environments  

RTX3090.

torch 1.10.1+cuda 11.2.

python 3.6.13.

numpy 1.23.4.

scipy 1.5.4.

scikit-learn 0.23.2.

We suggest you create a new environment with `conda create -n MCRec python=3.6`
And then conduct: `pip install -r requirements.txt`

## Quick Start

``````python
$ python main.py -- dataset Dunnhumby_sample
``````

## Running the code

create folder `./src/all_results/Dunnhumby`

```python
$ python main.py --dataset Dunnhumby --lr 0.0001 --l2 0.01 --Lh 9 --Lv 7
$ python main.py --dataset Instacart --lr 0.0001 --l2 0.0001 --Lh 6 --Lv 2
$ python main.py --dataset ValuedShopper --lr 0.01 --l2 0.0001 --Lh 5 --Lv 5
```
