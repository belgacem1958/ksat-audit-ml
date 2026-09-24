"""Model factories used across the analyses.

The reference model is a random forest (300 trees, min_samples_leaf=3,
random_state=0). Four model classes are compared under identical protocols:
random forest, histogram gradient boosting, SVM-RBF and linear regression.
"""
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

from config import RF_PARAMS


def rf(**overrides):
    """Random forest with the paper's reference hyperparameters."""
    params = dict(RF_PARAMS)
    params.update(overrides)
    return RandomForestRegressor(**params)


def model_zoo():
    """The four model classes compared in the paper."""
    return {
        "RandomForest": rf(),
        "HistGB": HistGradientBoostingRegressor(random_state=0, max_iter=300),
        "SVM-RBF": SVR(C=10.0, gamma="scale", epsilon=0.1),
        "Linear": LinearRegression(),
    }