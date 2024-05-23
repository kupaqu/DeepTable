import numpy as np

from pymfe.mfe import MFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.base import clone

def get_metafeatures_vector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mfe = MFE(groups=['model-based'])
    mfe.fit(X, y, suppress_warnings=True)
    ft = np.array(mfe.extract(suppress_warnings=True)[1])
    ft[np.isnan(ft)] = 0
    metafeatures_vector = MinMaxScaler().fit_transform(ft.reshape(-1, 1)).flatten()

    return metafeatures_vector

def fit_evaluate(clf, X: np.ndarray, y: np.ndarray) -> float:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

def get_lambda_vector(X: np.ndarray, y: np.ndarray, clfs: list) -> np.ndarray:
    scores = np.array([fit_evaluate(clone(clf), X, y) for clf in clfs])
    lambda_vector = np.exp(scores)/sum(np.exp(scores)) # softmax

    return lambda_vector