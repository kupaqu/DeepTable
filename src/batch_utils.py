import numpy as np
import torch

from typing import List, Tuple

from utils import get_lambda_vector, get_metafeatures_vector

# this module is used only with generated tables,
# which have column dimension before rows
# so we need to permute these dimensions,
# before passing it in .utils functions.

def preprocess_batch(X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.permute(X, (0, 2, 1))
    X, y = X.numpy(force=True), y.numpy(force=True)

    return X, y

def get_batch_lambda(X: torch.Tensor, y: torch.Tensor, clfs: List) -> torch.Tensor:
    X, y = preprocess_batch(X, y)
    batch_lambda = []
    for i in range(X.shape[0]):
        batch_lambda.append(get_lambda_vector(X[i], y[i], clfs))

    return torch.Tensor(batch_lambda)

def get_batch_metafeatures(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    X, y = preprocess_batch(X, y)
    batch_meta = []
    for i in range(X.shape[0]):
        batch_meta.append(get_metafeatures_vector(X[i], y[i]))

    return torch.Tensor(batch_meta)