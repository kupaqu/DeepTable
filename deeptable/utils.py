import numpy as np

from typing import Dict, List
from pymfe.mfe import MFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.base import clone, ClassifierMixin

def get_metafeatures_vector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Get metafeatures from table (X) and target column (y).

    Firstly, this function extracts model-based metafeatures
    vector, after that scaling it with MixMaxSclaer.

    Args:
        X (np.ndarray): Table with shape (n_rows, n_cols)
        y (np.ndarray): Target column with shape (n_rows)

    Returns:
        np.ndarray: Model-based metafeatures vector
    """
    mfe = MFE(groups=['model-based'])
    mfe.fit(X, y, suppress_warnings=True)
    ft = np.array(mfe.extract(suppress_warnings=True)[1])
    ft[np.isnan(ft)] = 0
    metafeatures_vector = MinMaxScaler().fit_transform(ft.reshape(-1, 1)).flatten()

    return metafeatures_vector

def fit_evaluate(clf: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> float:
    """Fit and evaluate accuracy sklearn classifier.

    Args:
        clf (ClassifierMixin): Classifier object
        X (np.ndarray): Table with shape (n_rows, n_cols)
        y (np.ndarray): Target column with shape (n_rows)

    Returns:
        float: Accuracy score
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

def get_lambda_vector(X: np.ndarray, y: np.ndarray, clfs: List[ClassifierMixin]) -> np.ndarray:
    """Calculate lambda vector which contains classifiers perfomance on dataset.

    Args:
        X (np.ndarray): Table with shape (n_rows, n_cols)
        y (np.ndarray): Target column with shape (n_rows)
        clfs (List[ClassifierMixin]): Classifiers

    Returns:
        np.ndarray: Lambda vector
    """
    scores = np.array([fit_evaluate(clone(clf), X, y) for clf in clfs])
    lambda_vector = np.exp(scores)/sum(np.exp(scores)) # softmax

    return lambda_vector

def join_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Join two dictionaries keys and values.

    Args:
        dicts (List[Dict[str, float]]): List of dictionaries

    Returns:
        Dict[str, float]: Joined dictionary
    """
    output_dict = {}
    for dict_ in dicts:
        output_dict.update(dict_)

    return output_dict

def sum_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Sum values of dictionaries.

    Args:
        dicts (List[Dict[str, float]]): Dictionaries with same keys

    Returns:
        Dict[str, float]: Dictionaries with summarized values
    """
    output_dict = dicts[0].copy()
    for dict_ in dicts[1:]:
        for key, value in dict_.items():
            output_dict[key] += value
    
    return output_dict