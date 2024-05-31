import torch
import torch.nn.functional as F

from .deepsets import *

class DeepTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.equiv_0 = EquivLinear(1, 64)
        self.inv_0 = InvLinear(64, 128)
        self.equiv_1 = EquivLinear(128, 256)
        self.inv_1 = InvLinear(256, 512)

    def forward(self, x):
        N = x.shape[1] # количество столбцов в таблице
        x = x.unsqueeze(-1) # (batch_size, N, M, 1)
        x = x.flatten(0, 1) # (batch_size * N, M, 1)

        x = self.equiv_0(x) # (batch_size * N, M, 64)
        x = F.relu(x)
        x = self.inv_0(x) # (batch_size * N, 128)
        x = x.reshape(-1, N, 128) # (batch_size, N, 128)

        x = self.equiv_1(x) # (batch_size, N, 256)
        x = F.relu(x)
        x = self.inv_1(x) # (batch_size, 512)

        return x