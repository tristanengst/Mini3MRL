# Mini Masked Multi-Modal Representation Learning
Code for the toy example.

A lot of what's here is taken from [this repo](https://github.com/ReyhaneAskari/pytorch_experiments/blob/master/DAE.py). Many thanks!

## Setup
In a conda environment:
```
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge wandb tqdm scikit-learn matplotlib plotly pandas
pip install kaleido
```

## Usage
Run an IMLE model:
```
Python IMLE_DAE.py --code_bs=60000 --epochs=20 --evals=20 --ipe=50 --lr=0.001 --ns=128 --seed=5000 --std=0.8 --suffix=IMLE --wandb=disabled
```
Run a plain DAE model:
```
python DAE.py --epochs=1000 --evals=20 --lr=0.001 --std=0.8 --suffix=DAE --wandb=disabled
```
