"""
Module for training population models and predicting population.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copy import deepcopy

import pandas as pd
from forestci import random_forest_error
from sklearn.preprocessing import StandardScaler

# local imports
from population.models import get_model
from population.utilities import write_raster
from population.visualization import *
from population.scoring import aggregate_percent_error
from population.visualization import prediction_error


def pop_histogram(y, ax, y_label):
    """ Plot histogram of population """
    ax.hist(y, 50, (0, 100))
    ax.set_xlabel('Population')
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 20)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")


def get_split(df, k, verbose=True):
    """ Return training/validation split for fold k in df """
    df_train = df[df['fold'] != k].drop('fold', axis=1)
    df_val = df[df['fold'] == k].drop('fold', axis=1)
    ratio = len(df_val) / (len(df_val) + len(df_train))
    if verbose:
        print(
            f'Training on {len(df_train)} samples, validating on {len(df_val)}, {(1 - ratio) * 100:.0f}/{ratio * 100:.0f} split')
    return df_train, df_val


def get_metrics(y_true, y_pred):
    """ return metrics computed on observed and predicted values formatted for table """
    return [f'{r2_score(y_true, y_pred):0.2f}',
            f'{meape(y_true, y_pred) * 100:0.1f}%'.zfill(5),
            f'{ameape(y_true, y_pred):0.2f}',
            f'{median_absolute_error(y_true, y_pred):0.2f}',
            f'{aggregate_percent_error(y_true, y_pred) * 100:0.1f}%'.zfill(5)]


def cross_val(reg_master, df, features, target, return_models=True, log=False):
    """
    Train regression model using cross-validation on dataframe pre-split into folds.

    Args:
            reg_master (sklearn.model_selection.GridSearchCV): A grid search instance to be trained.
            df (pd.DataFrame): The dataframe used for training and validation.
            features (:obj:`list` of :obj:`str`): The dataframe columns used as features during training.
            target (str): The dataframe column used as target variable during training.
            return_models (:obj:`bool`, optional): Whether to return models. Defaults to False.
            log (:obj:`bool`, optional): Whether to predict predict log of target. Defaults to False.

    Returns:
        (:obj:`np.ndarray`, :obj:`np.ndarray`, :obj:`list` of :obj:`model`): Model predictions for each row of the
        dataframe, model variance for each row of the dataframe, and list of model trained on each cross validation fold
        returned if return_models is True.
    """
    ks = np.sort(df['fold'].unique())  # list of folds specified in dataframe
    if len(ks) == 0:
        print("No folds specified in dataframe")
        return

    y_pred = []
    y_var = []
    models = []
    scaler = StandardScaler()
    X = df[features].to_numpy()
    scaler.fit(X)
    # sys.exit()
    for i in ks:  # iterate through folds
        # start with fresh grid search instance 
        reg = deepcopy(reg_master)

        # get train/val split for this fold
        df_train, df_val = get_split(df, i)

        # convert dataframes to numpy arrays
        X_train = df_train[features].to_numpy()
        y_train = df_train[target].to_numpy().ravel()
        X_val = df_val[features].to_numpy()
        y_val = df_val[target].to_numpy().ravel()

        if log:
            y_train = np.log(y_train)
            y_val = np.log(y_val)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        # fit model with grid search
        gs = reg.fit(X_train, y_train)
        model = gs.best_estimator_
        print(reg.best_params_)
        y_pred += list(model.predict(X_val))

        if type(model).__name__ == 'RandomForestRegressor': # TODO: variance workaround could be cleaner
            y_var += list(random_forest_error(model, X_train, X_val))
        else:
            y_var += [0 for _ in range(len(y_val))]

        models.append(model)
    if log:
        y_pred = np.exp(y_pred)
    if return_models:
        return np.array(y_pred), np.array(y_var), models  # TODO: bad practice to vary return type
    else:
        return np.array(y_pred), np.array(y_var)


def run_experiment(df, features, cvs, logs, model_names, target='pop'):
    """
    Run experiment by training given models with specified feature set. Results are saved to dataframe.

    Args:
            df (pd.DataFrame): The dataframe used for training and validation.
            features (:obj:`list` of :obj:`str`): The dataframe columns used as features during training.
            cvs (:obj:`list` of :obj:`sklearn.model_selection.GridSearchCV`): Grid search instance for each model.
            logs (:obj:`list` of :obj:`bool`): Whether to predict predict log of target for each model.
            model_names (:obj:`list` of :obj:`str`): Name of each model.
            target (:obj:`str`, optional): The dataframe column used as target variable during training. Defaults to 'pop'.

    Returns:
            :obj:`list` of :obj:`sklearn.base.BaseEstimator`: List of models trained on df via cross validation.
    """
    models = []
    for cv, log, name in zip(cvs, logs, model_names):  # run experiment using each model
        print(f'\nTraining {name}\n')
        y_pred, y_var, model = cross_val(cv, df, features, target, log=log, return_models=True)
        df[name] = y_pred
        df[f'{name}_var'] = y_var
        models.append(model)
    return models


def plot_model(model, ax, df, features, model_name, out_dir, plot_full):
    """ Plot prediction error and feature importance for model. """
    # plot error
    print('Plotting prediction error\n')
    prediction_error(df, true='pop', pred=model_name, ax=ax, color=True)
    ax.set_title(model_name)
    # plot importance
    print('Plotting feature importance\n')
    cs = get_colors(features.shape[0])
    f, ax = feature_importance(model, features, cs, crop=True, n_show=15)
    f.tight_layout(pad=1.2)
    f.savefig(out_dir + f'{model_name}_importance.pdf', bbox_inches='tight')
    if plot_full:  # warning: can be slow and cramped
        f, ax = feature_importance(model, features, cs, crop=False)
        f.tight_layout(pad=1.2)
        f.savefig(out_dir + f'{model_name}_importance_full.pdf', bbox_inches='tight')


def run_experiments(df_master, cvs, model_names, logs, features_list, out_dir_list, experiment_dir, plot_full=True,
                    ignore_outliers_val=True, ignore_zeros_val=True, target='pop'):
    """
    Args:
        df_master (:obj:`pd.DataFrame`): The dataframe used to run the experiments.
        cvs (:obj:`list` of :obj:`sklearn.model_selection.GridSearchCV`): Grid search instance for each model.
        model_names (:obj:`list` of :obj:`str`): Name of each model.
        logs (:obj:`list` of :obj:`bool`): Whether to predict predict log of target for each model.
        features_list (:obj:`list` of :obj:`str`): Dataframe columns to use as features in each experiment.
        out_dir_list (:obj:`list` of :obj:`str`): Directories to output results of each experiment.
        experiment_dir: Parent directory for experiment directories.
        plot_full (:obj:`bool`, optional): Whether or not to plot feature importance with full feature set.
            Avoid for large feature sets. Defaults to True.
        ignore_outliers_val (:obj:`bool`, optional): Whether to always ignore outlier tiles during validation.
            Defaults to True.
        ignore_zeros_val (:obj:`bool`, optional): Whether to always ignore zero population tiles during validation.
            Defaults to True.
        target (:obj:`str`, optional): The dataframe column used as target variable during training. Defaults to 'pop'.

    Returns:
        None.
    """
    n_models = len(model_names)  # number of models
    n_features = len(features_list)  # number of feature sets to run each model on
    n_metrics = 5  # number of metrics to report in results table
    results = np.zeros((n_models * n_features, n_metrics * 2), dtype='object')  # results table
    for i, features in enumerate(features_list):  # iterate through feature sets
        out_dir_master = out_dir_list[i]  # sub-directory for results for this feature set
        for j, include_outliers in enumerate([True, False]):  # include/exclude outliers
            inc = 'included' if include_outliers else 'removed'
            out_dir = out_dir_master + f'/outliers_{inc}/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            df = deepcopy(df_master)
            df = df[(df['outlier'] == include_outliers) | (df['outlier'] == False)]

            models = run_experiment(df, features, cvs, logs, model_names)

            if ignore_outliers_val:
                df = df[df['outlier'] == False]  # validate without outliers
            if ignore_zeros_val:
                df = df[df[target] >= 1]  # validate without injected zero examples

            y_true = df[target].to_numpy(copy=True)  # get true values

            # update results for each model and plot prediction error, feature importance
            f_error, axarr_error = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3))
            if n_models == 1:
                axarr_error = [axarr_error]
            for k, model_name in enumerate(model_names):
                results_row = get_metrics(y_true, df[model_name])
                results[k * n_features + i, j * n_metrics:(j + 1) * n_metrics] = results_row
                plot_model(models[k], axarr_error[k], df, features, model_name, out_dir, plot_full)
            f_error.savefig(out_dir + 'prediction_error.pdf', bbox_inches='tight')
            f_error.savefig(out_dir + 'prediction_error.svg', bbox_inches='tight')
            df.to_csv(os.path.join(out_dir, 'estimates.csv'))

    df_results = pd.DataFrame(results)
    with open(os.path.join(experiment_dir, 'table.tex'), 'w') as f:
        f.write(df_results.to_latex())  # write table to file


# def get_features(csv_path):  # TODO: appears to be redundant
#     return np.loadtxt(csv_path, delimiter=',')

def expand_features(feature_sets, cols):
    """ Expand features of form <prefix>_# to cover all columns in cols where # represents a number. """
    for i in range(len(feature_sets)):
        for j in range(len(feature_sets[i])):
            if feature_sets[i][j].endswith('_#'):
                prefix = feature_sets[i][j][:-1]  # get suffix without number placeholder
                matching_features = [c for c in cols if c.startswith(prefix) and c[len(prefix):].isnumeric()]
                feature_sets[i].pop(j)
                feature_sets[i] += matching_features
    return feature_sets


def run_predictions(df, params, prng):
    """ Run experiments described in params over dataset df. """
    model_names = params['models']
    cvs = [get_model(model, prng) for model in model_names]
    logs = params['log']
    feature_sets = [list(np.loadtxt(f, dtype=str, ndmin=1, comments=None)) for f in params['feature_sets']]
    feature_sets = expand_features(feature_sets, list(df.columns))
    if params['show_roi']:
        feature_sets = [f.append('roi_num') for f in feature_sets]
    feature_sets = np.array(feature_sets)
    experiment_dir = params['experiment_dir']
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    out_dir_list = [os.path.join(experiment_dir, os.path.basename(f)[0:-4]) for f in params['feature_sets']]
    run_experiments(df, cvs, model_names, logs, feature_sets, out_dir_list, experiment_dir, plot_full=False)


def to_raster(df, target, shape):
    """ Convert df[target] to raster of specified shape. """
    raster = np.zeros(shape)
    for index, row in df.iterrows():
        x, y, n = row['x'], row['y'], row[target]
        raster[y, x] = n
    return raster


def run_estimation(df, df_full, params, prng):
    """ Run estimation across full roi(s) based on params. """
    model_name = params['model']
    cv = get_model(model_name, prng)
    log = params['log']
    rois = params['rois']
    features = np.loadtxt(params['feature_set'], dtype=str)
    prediction_dir = params['prediction_dir']
    survey_paths = params['pop_rasters']
    include_outliers = params['include_outliers']
    hurdle_feature = ''
    if 'hurdle_feature' in params:
        hurdle_feature = params['hurdle_feature']
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    df = df[(df['outlier'] == include_outliers) | (df['outlier'] == False)]

    print(f'Estimating population across {rois}')

    models = run_experiment(df, features, [cv], [log], [model_name])[0]  # TODO: ugly use of this function

    scaler = StandardScaler()  # TODO: do this in dataset building step?
    X_sub = df[features].to_numpy()  # use survey data to fit scaler
    scaler.fit(X_sub)
    X = scaler.transform(df_full[features].to_numpy())

    df_full[model_name] = np.zeros(X.shape[0])
    for model in models:  # average result from model for each cv fold TODO: this is sub-optimal, should train one model
        y_pred = model.predict(X)
        if log:
            y_pred = np.exp(y_pred)
        if hurdle_feature:
            y_pred[np.where(df_full[hurdle_feature].to_numpy() < 2)] = 0
        df_full[model_name] += y_pred
    df_full[model_name] /= len(models)
    print(df_full[model_name])
    for roi, survey_path in zip(rois, survey_paths):
        shape = imread(survey_path).shape
        raster = to_raster(df_full.loc[df_full['roi'] == roi], model_name, shape)
        # raster = np.expand_dims(raster,axis=-1)
        write_raster(raster, survey_path, os.path.join(prediction_dir, f'pop_pred_{roi}.tif'))


