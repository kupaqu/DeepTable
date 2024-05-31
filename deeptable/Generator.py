import torch

from torch import nn

class Generator(nn.Module):
    def __init__(self, n_metas: int):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(32+n_metas, 64, kernel_size=(4, 7)), # не добавляем паддинг и страйд т. к. делит на классы
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 128, kernel_size=(4, 7), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=(4, 7), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=(4, 7), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, kernel_size=(4, 8), stride=(1, 2), padding=(0, 2))
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor):
        concat = torch.cat((x, meta), 1)
        return self.upconv(concat)