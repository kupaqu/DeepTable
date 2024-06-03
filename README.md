# DeepTable Conditional GAN

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kV8jI4SqwD3VnUzXromRg-tMl5sshSax?usp=sharing)

Python implementation of Conditional GAN with discriminator based on DeepTable architecture which chooses most accurate classifier on particular dataset.

## What's coming next
- Multiclass datasets support
- Datasets with categorical and ordinal data

## Installation

Install using `pip`

For the latest development release:

``` bash
pip install git+https://github.com/kupaqu/DeepTable.git
```

## Usage

Initialize dataset instance

``` python
from deeptable import OpenMLDataset

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

clfs = [LogisticRegression(max_iter=1000),
        GaussianNB(),
        RandomForestClassifier()]

train_dataset = OpenMLDataset('path/to/train/dataset/directory')
test_dataset = OpenMLDataset('path/to/test/dataset/directory')
```

Dataset directory structure should be like this:

```
data_dir
└─── data_1
│    └───zero.csv   # negative class
│    └───one.csv    # positive class
└─── data_2
     └───zero.csv
│    └───one.csv
└─── ...
```

Initialize trainer and train a model

``` python
trainer = Trainer(clfs=clfs,
                  train_dataset=train_dataset,
                  test_dataset=test_dataset,
                  run_dir='path/to/run/directory')

trainer.to('cuda')

trainer.train(epochs=10,
              test_epoch=2,
              track_metric='Lambda classifier F1 macro')
```

Load checkpoint to existing trainer instance to continue training

``` python
import torch

trainer.load_checkpoint(path='path/to/checkpoint.pt')
```

Load checkpoint and extract GAN

``` python
from deeptable import GAN

gan = GAN(n_clfs=3, n_metas=97)
gan.d.load_state_dict(checkpoint['model_state_dict']['d'])
gan.g.load_state_dict(checkpoint['model_state_dict']['g'])

```

## Acknowledgement and References

This project is based on research and code from several papers and open-source repositories.

All deep learning execution is based on [Pytorch](https://pytorch.org).

Deep Table are based on experiments in this [repository](https://github.com/garipovroma/bachelor-thesis).

Deep Table uses the Deep Sets layers from this [repository](https://github.com/dpernes/deepsets-digitsum) which described in Deep Sets by Zaheer et al. [paper](https://arxiv.org/abs/1703.06114).