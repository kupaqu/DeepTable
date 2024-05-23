import torch

from torch import nn

from DeepTable import DeepTable

class Discriminator(nn.Module):
    def __init__(self, n_clfs: int, n_metas: int):
        super().__init__()
        self.deeptable = DeepTable()
        self.metaclassifier = nn.Sequential(
            nn.Linear(512+n_metas, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, n_clfs),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(512+n_metas, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor):
        features = self.deeptable(x)
        concat = torch.cat((features, meta), 1)

        lambda_ = self.metaclassifier(concat) # выдает лучший алгоритм на заданном датасете
        label = self.discriminator(concat) # метка датасета (реальный он или сгенерирован генеративной моделью)

        return lambda_, label
