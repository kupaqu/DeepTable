import torch
import os
import pandas as pd

from typing import List

from sklearn.base import ClassifierMixin
from utils import get_lambda_vector, get_metafeatures_vector

class OpenMLDataset(torch.utils.data.Dataset):
    def __init__(self, clfs: List[ClassifierMixin], data_dir: str, test: bool):
        self.clfs = clfs
        self.tables, self.targets, \
              self.lambdas, self.metas = self._load_dataset(data_dir, test)

    def _load_dataset(self, data_dir: str, test: bool):
        tables = []
        targets = []
        lambdas = []
        metas = []
        
        test_dirs = [] # contains filenames of test csv's
        with open(os.path.join(data_dir, 'test.txt')) as f:
            for line in f.read().splitlines():
                test_dirs.append(line)
        
        csvs_dir = os.path.join(data_dir, 'csv')
        for csv_dir in os.listdir(csvs_dir):
            if not test and csv_dir in test_dirs \
                or test and csv_dir not in test_dirs:
                continue

            zero_table = pd.read_csv(os.path.join(csvs_dir, csv_dir, 'zero.csv'), header=None) # rows which target is 0
            one_table = pd.read_csv(os.path.join(csvs_dir, csv_dir, 'one.csv'), header=None) # rows which target is 1
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
    
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier

# if __name__ == '__main__':

#     train_dataset = OpenMLDataset(clfs=[LogisticRegression(max_iter=1000),
#                                         GaussianNB(),
#                                         RandomForestClassifier()],
#                                   data_dir='data',
#                                   test=True)
    
#     print(len(train_dataset.tables), len(train_dataset.targets), len(train_dataset.lambdas), len(train_dataset.metas))