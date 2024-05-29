import torch
import os
import pandas as pd

from typing import List
from tqdm import tqdm

from sklearn.base import ClassifierMixin
from utils import get_lambda_vector, get_metafeatures_vector

class OpenMLDataset(torch.utils.data.Dataset):
    def __init__(self, clfs: List[ClassifierMixin], data_dir: str, test: bool):
        self.clfs = clfs
        self.tables, self.targets, \
              self.lambdas, self.metas = self._preprocess_dataset(data_dir, test)

    def _preprocess_dataset(self, data_dir: str):
        tables = []
        targets = []
        lambdas = []
        metas = []
        
        print(f'Loading dataset from {data_dir}')
        for csv_dir in tqdm(os.listdir(data_dir)):
            zero_table = pd.read_csv(os.path.join(data_dir, csv_dir, 'zero.csv'), header=None) # rows which target is 0
            one_table = pd.read_csv(os.path.join(data_dir, csv_dir, 'one.csv'), header=None) # rows which target is 1
            table = torch.tensor(pd.concat([zero_table, one_table], axis=0).values, dtype=torch.float32)

            zero_target = torch.zeros((len(zero_table),))
            one_target = torch.ones((len(one_table),))
            target = torch.cat((zero_target, one_target), dim=0)

            lambda_ = torch.tensor(get_lambda_vector(table.numpy(), target.numpy(), self.clfs), dtype=torch.float32)
            meta = torch.tensor(get_metafeatures_vector(table.numpy(), target.numpy()), dtype=torch.float32)

            # zero dimension should be columns before passing inside DeepTable,
            # do this after calculation of lambda and meta vectors 
            table = torch.permute(table, (1, 0))

            tables.append(table)
            targets.append(target)
            lambdas.append(lambda_)
            metas.append(meta)

        return tables, targets, lambdas, metas
    
    def __len__(self):
        return len(self.tables)
    
    def __getitem__(self, i):
        return self.tables[i], self.targets[i], \
            self.lambdas[i], self.metas[i]