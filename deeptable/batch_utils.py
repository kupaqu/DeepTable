import numpy as np
import torch

from typing import List, Tuple

from .utils import get_lambda_vector, get_metafeatures_vector

# this module is used only with generated tables,
# which have column dimension before rows
# so we need to permute these dimensions,
# before passing it in .utils functions.

def preprocess_batch(X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess batch X, y to pass into utils functions.

    Args:
        X (torch.Tensor): Batch of tables with shape (batch_size, n_cols, n_rows)
        y (torch.Tensor): Batch of targets with shape (batch_size, n_rows)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Processed X, y
    """
    if y is None: # if y is None, it means that table is generated, so assume zeros comes before ones
        batch_size = X.shape[0]
        n_rows = X.shape[2]
        y = torch.cat([torch.zeros((batch_size, n_rows//2)), torch.ones((batch_size, n_rows//2))], dim=1)

    X = torch.permute(X, (0, 2, 1))
    X, y = X.numpy(force=True), y.numpy(force=True)

    return X, y

def get_batch_lambda(clfs: List, X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """Calculate batch lambda vector which contains classifiers perfomance on dataset.

    Args:
        X (np.ndarray): Batch of tables with shape (batch_size, n_cols, n_rows)
        y (np.ndarray): Batch of targets with shape (batch_size, n_rows)
        clfs (List[ClassifierMixin]): Classifiers

    Returns:
        np.ndarray: Batch lambda vector
    """
    X, y = preprocess_batch(X, y)
    batch_lambda = []
    for i in range(X.shape[0]):
        batch_lambda.append(get_lambda_vector(X[i], y[i], clfs))

    return torch.Tensor(batch_lambda)

def get_batch_metafeatures(X: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """Get batch metafeatures from table (X) and target column (y).

    Args:
        X (np.ndarray): Batch of tables with shape (batch_size, n_cols, n_rows)
        y (np.ndarray): Batch of targets with shape (batch_size, n_rows)

    Returns:
        np.ndarray: Model-based batch metafeatures vector.
    """
    X, y = preprocess_batch(X, y)
    batch_meta = []
    for i in range(X.shape[0]):
        batch_meta.append(get_metafeatures_vector(X[i], y[i]))

    return torch.Tensor(batch_meta)