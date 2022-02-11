"""
Module containing functions for building population datasets.
"""

import numpy as np
import pandas as pd
import csv
import os
import pickle
from tqdm import tqdm
from tifffile import imread
from copy import deepcopy


def in_bounds(raster, y, x):
    """ Check if array access at index (y,x) is in bounds of raster. """
    return 0 <= y < raster.shape[0] and 0 <= x < raster.shape[1]


def get_context(raster, y, x, n):
    """ Return mean of n x n context area surrounding index (y,x) of raster. """
    def get_range(c):
        return range(c + (1 - n) // 2, c + (1 + n) // 2)
    vals = [raster[i, j]
        for i in get_range(y) for j in get_range(x) if in_bounds(raster, i, j)]
    return np.mean(np.array(vals), axis=0)


def get_val_split(df, n=2, coord='y', leaf=False):
    """
    Returns list of (n x n) pandas dataframes, corresponding to splitting df
    spatially into (n x n) segments with approx equal numbers of survey points
    by:
        1. splitting into n segments by y coordinate, then
        2. splitting each segment into n segments by x coordinate.
    """
    df = deepcopy(df)
    df_sorted = df.sort_values(by=coord)
    increment = len(df_sorted) // n
    coords = [df_sorted[coord].to_numpy()[i]
        for i in range(0, len(df_sorted) - increment + 1, increment)]
    coords.append(max(df[coord]))
    dfs = []
    for i in range(n - 1):
        dfs.append(df[(coords[i] <= df[coord]) & (df[coord] < coords[i + 1])])
    dfs.append(df[(coords[n - 1] <= df[coord]) & (df[coord] <= coords[n])])
    if not leaf:
        dfs = np.array([get_val_split(d, n=n, coord='x', leaf=True)
                        for d in dfs], dtype=object).flatten()
    return dfs


def label_folds(dfs):
    """ Returns concatenatenation of dataframes in list dfs, each labelled with
        a unique value in the 'fold' column. """
    df_combined = pd.DataFrame()
    for i, df in enumerate(dfs):
        df['fold'] = i
        df_combined = df_combined.append(df, ignore_index=True)
    return df_combined


def mark_outliers(df, outliers, col=2):
    """ Mark members of df with outlier values defined in csv file. """
    df['outlier'] = False
    with open(outliers) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # skip header
        for row in csv_reader:
            reject = int(row[col])
            if reject == 0:
                x, y = int(row[0]), int(row[1])
                df.loc[(df['x'] == x) & (df['y'] == y), 'outlier'] = True
    return df


def init_cols(feature_names, context_sizes):
    """ Initialize dictionary with column names. """
    d = {'x': [], 'y': []}
    for f in feature_names:
        d[f] = []
        for size in context_sizes:
            d[f'{f}_context_{size}x{size}'] = []
    d['pop'] = []
    return d


def build_row(d, y, x, pop, rasters, feature_names, context_sizes):
    """
    Build row (in-place) of population dataset corresponding to grid cell .

    Args:
        d (dict): Dictionary used to build dataset.
        y (int): y coord of grid cell covered by row.
        x (int): x coord of grid cell covered by row.
        pop (int): population of grid cell (where known).
        rasters (:obj:`list` of :obj:`np.ndarray`): rasters containing features.
        feature_names (:obj:`list` of :obj:`np.ndarray`): name for each raster
            in rasters.
        context_sizes (:obj:`list` of :obj:`int`): sizes of n x n feature
            contexts to compute.

    Returns:
        None.
    """
    d['y'].append(y)
    d['x'].append(x)
    for r, f in zip(rasters, feature_names):
        d[f].append(r[y, x])
    for size in context_sizes:
        for r, f in zip(rasters, feature_names):
            d[f'{f}_context_{size}x{size}'].append(get_context(r, y, x, size))
    d['pop'].append(pop)


def build_dataset(params, survey_only=True):
    """
    Build dataset of features with context, population labels according to
    specifications in file params_path.

    Args:
            params (dict): A dictionary holding parameters.
            survey_only (bool): True if building dataset for only sampled survey
                tiles.

    Returns:
            pd.DataFrame: Pandas DataFrame containing full dataset.
    """
    # parse params
    in_dir = params['input_dir']
    rois = params['rois']
    pop_rasters = params['pop_rasters']
    outliers_paths = params['outliers_paths']
    context_sizes = params['context_sizes']
    if 'zero_label_paths' in params:
        zero_labels = params['zero_label_paths']
    else:
        zero_labels = ['' for _ in rois]
    # Construct dataset.
    dfs = []
    for roi, pop, outliers_path, zero_labels_path in zip(
            rois, pop_rasters, outliers_paths, zero_labels):
        raster_dir = os.path.join(in_dir, roi)
        files = os.listdir(raster_dir)
        rasters = [imread(os.path.join(raster_dir, f))
                   for f in files if f.endswith('.tif')]
        feature_names = [os.path.splitext(f)[0]
                         for f in files if f.endswith('.tif')]
        pop = imread(pop)
        zero_set = set()
        # Skip if no zero label path provided for this roi.
        if zero_labels_path:
            with open(zero_labels_path, 'rb') as file:
                zero_set = pickle.load(file)
        # Initialize dictionary with column names.
        d = init_cols(feature_names, context_sizes)
        print(f'Building dataset for {roi}')
        for y in tqdm(range(pop.shape[0])):
            for x in range(pop.shape[1]):
                in_zero = (y, x) in zero_set
                n = pop[y, x]  # Population of cell.
                if in_zero:
                    n = 1e-4  # TODO (isaac): Messy workaround for log.
                # Populate row with raster values at cell.
                if n > 0 or in_zero or not survey_only:
                    build_row(d, y, x, n, rasters, feature_names, context_sizes)
        df = pd.DataFrame(d)
        df['x'] = df['x'].astype(int)
        df['y'] = df['y'].astype(int)
        df['roi'] = roi
        if survey_only:  # Cross_val folds only relevant for survey data.
            if len(zero_set) > 0:
                df_survey = label_folds(get_val_split(df[df['pop'] >= 1]))
                df_zero = label_folds(get_val_split(df[df['pop'] < 1]))
                df = df_survey.append(df_zero, ignore_index=True)
            else:
                df = label_folds(get_val_split(df))
            df = mark_outliers(df, outliers_path)
        else:  # Population irrelevant for non-survey data.
            df = df.drop(labels='pop', axis=1)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if survey_only:
        df = df.sort_values(by='fold', ascending=True)
    df['roi'] = pd.Categorical(df['roi'])  # Set to categorical to allow codes.
    df['roi_num'] = df['roi'].cat.codes  # Codes that can be used in modelling.
    df = df.reset_index(drop=True)
    return df
