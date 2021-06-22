"""
Module containing getters for untrained population models.
"""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Lasso
from sklearn.metrics import make_scorer, median_absolute_error
from sklearn.model_selection import GridSearchCV


def get_poisson():
    """ Return grid search instance for Poisson regression model. """
    pr = PoissonRegressor(max_iter=200)
    reg_pr = GridSearchCV(pr, {'alpha': np.linspace(0.1, 1, 50)},
                          scoring=make_scorer(median_absolute_error, greater_is_better=False), cv=3, n_jobs=-1)
    return reg_pr


def get_lasso():
    """ Return grid search instance for lasso regression model. """
    lr = Lasso(max_iter=1000)
    reg_lr = GridSearchCV(lr, {'alpha': np.linspace(0.1, 1, 50)},
                          scoring=make_scorer(median_absolute_error, greater_is_better=False), cv=3,
                          n_jobs=-1)
    return reg_lr


def get_dummy():
    """ Return grid search instance for dummy regression model (returns mean of training set). """
    dr = DummyRegressor(strategy='mean')
    reg_dr = GridSearchCV(dr, {}, scoring=make_scorer(median_absolute_error, greater_is_better=False), cv=3, n_jobs=-1)
    return reg_dr


def get_rf(prng):
    """
    Get grid search instance for Random Forest regression model.

    Args:
        prng (:obj:`np.random.RandomState`): Random state used to construct forest.

    Returns:
        sklearn.model_selection.GridSearchCV: Random Forest grid search instance.
    """
    rf = RandomForestRegressor(n_jobs=-1, random_state=prng)
    param_grid = {'n_estimators': list(range(100, 500, 100)),
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [1, 2]}
    reg_rf = GridSearchCV(rf, param_grid, scoring=make_scorer(median_absolute_error, greater_is_better=False), cv=3,
                          verbose=0, n_jobs=-1)
    return reg_rf


def get_model(model_name, prng):
    """
    Get grid search instance for specified regression model.

    Args:
        model_name (str): Name of regression model.
        prng (:obj:`np.random.RandomState`): Random state used for stochastic models.

    Returns:
        sklearn.model_selection.GridSearchCV: Model grid search instance.
    """
    if model_name == 'poisson':
        return get_poisson()
    elif model_name == 'rf':
        return get_rf(prng)
    elif model_name == 'lasso':
        return get_lasso()
    elif model_name == 'dummy':
        return get_dummy()
    else:
        print('Unidentified model, check pipeline config')
