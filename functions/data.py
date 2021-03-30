import numpy as np
import pandas as pd
import csv
import os
import json
from tifffile import imread
from copy import deepcopy
from osgeo import gdal

# raster_path = './data/100m/'
# survey_path = './data/pop/'

def in_bounds(raster,y,x):
    ''' Check if array access at index (y,x) is in bounds of raster '''
    return 0 <= y < raster.shape[0] and 0 <= x < raster.shape[1]

def get_context(raster,y,x,n):
    ''' Return mean of n x n bounds-checked context area surrounding index (y,x) of raster '''
    vals = [raster[i,j] for i in range(y+(1-n)//2,y+(1+n)//2) 
            for j in range(x+(1-n)//2,x+(1+n)//2) if in_bounds(raster,i,j)]
    return np.mean(np.array(vals), axis=0)

def get_val_split(df,n=2,coord='y',leaf=False):
    '''
    Returns list of (n x n) pandas dataframes, corresponding to splitting df spatially into (n x n)
    segments with approx equal numbers of survey points by:
        1. splitting into n segments by y coordinate, then
        2. splitting each segment into n segments by x coordinate
    '''
    df = deepcopy(df)
    df_sorted = df.sort_values(by=coord)
    increment = len(df_sorted) // (n)
    coords = [df_sorted[coord].to_numpy()[i] for i in range(0,len(df_sorted)-increment+1,increment)]
    coords.append(max(df[coord]))
    dfs = []
    for i in range(n-1):
            dfs.append(df[(coords[i] <= df[coord]) & (df[coord] < coords[i+1])])
    dfs.append(df[(coords[n-1] <= df[coord]) & (df[coord] <= coords[n])])
    if not leaf:
        dfs = np.array([get_val_split(d,n=n,coord='x',leaf=True) for d in dfs],dtype=object).flatten()
    return dfs

def label_folds(dfs):
    ''' Returns concatenatenation of dataframes in list dfs, each labelled with a unique value in the 'fold' column '''
    df_combined = pd.DataFrame()
    for i,df in enumerate(dfs):
        df['fold'] = i
        df_combined = df_combined.append(df,ignore_index=True)
    return df_combined

def remove_outliers(df,outliers,strong=True,weak=True):
    ''' Return dataframe with outliers defined in csv file removed '''
    with open(outliers) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip header
        for row in csv_reader:
            reject = int(row[5])
            if reject > 0:
                if (reject == 1 and weak) or (reject==2 and strong):
                    x,y = int(row[0]), int(row[1])
                    df = df[(df['x'] != x) | (df['y'] != y)]
    return df

def mark_outliers(df,outliers,strong=True,weak=True,col=5): # TODO: remove?
    ''' Return dataframe with outliers defined in csv file marked '''
    df['outlier'] = False
    with open(outliers) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip header
        for row in csv_reader:
            reject = int(row[col])
            if reject > 0:
                if (reject == 1 and weak) or (reject==2 and strong):
                    x,y = int(row[0]), int(row[1])
                    df.loc[(df['x'] == x) & (df['y'] == y),'outlier'] = True
    return df

def mark_outliers_new(df,outliers,col=2):
    ''' Return dataframe with outliers defined in csv file marked (updated) '''
    df['outlier'] = False
    with open(outliers) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip header
        for row in csv_reader:
            reject = int(row[col])
            if reject == 0:
                x,y = int(row[0]), int(row[1])
                df.loc[(df['x'] == x) & (df['y'] == y),'outlier'] = True
    return df

def build_dataset(params): # construct dataframe from rasters, survey
    '''
    Build dataset of features with context, population labels according to 
    specifications in file params_path.
    
            Args:
                    params (dict): A dictionary holding parameters.

            Returns:
                    df (DataFrame): Pandas DataFrame containing full dataset.
    '''
    # parse params
    in_dir = params['input_dir']
    rois = params['rois']
    pop_rasters = params['pop_rasters']
    outliers_paths = params['outliers_paths']
    context_sizes = params['context_sizes']
    
    # construct dataset
    dfs = []
    for roi,pop,outliers_path in zip(rois,pop_rasters,outliers_paths):
        raster_dir = os.path.join(in_dir,roi)
        files = os.listdir(raster_dir)
        rasters = [imread(os.path.join(raster_dir,f)) for f in files if f.endswith('.tif')]
        feature_names = [os.path.splitext(f)[0] for f in files if f.endswith('.tif')]
        pop = imread(pop)
        
        d = {'x':[],'y':[]} # coordinates used to perform validation split spatially
        for f in feature_names:
            d[f] = []
        for size in context_sizes:
            for f in feature_names:
                d[f'{f}_context_{size}x{size}'] = []
        d['pop'] = []
        df = pd.DataFrame(d)
        count = 0
        for i in range(pop.shape[0]):
            for j in range(pop.shape[1]):
                n = pop[i,j] # population of cell
                if n > 0:
                    # populate row with raster values at cell
                    row = np.array([j,i])
                    for r in rasters:
                        row = np.append(row,r[i,j])
                    for size in context_sizes:
                        for r in rasters:
                            row = np.append(row,get_context(r,i,j,size))
                            #print(row.shape[0])
                    row = np.append(row,n)
                    #print(row.shape)
                    df.loc[count] = row
                    count+=1
        df['x'] = df['x'].astype(int)
        df['y'] = df['y'].astype(int)
        df['roi'] = roi
        df = label_folds(get_val_split(df))
        df = mark_outliers_new(df,outliers_path)
        dfs.append(df)
    df = dfs[0]
    for df_i in dfs[1:]: # combine dataframes for each roi
        df = df.append(df_i,ignore_index=True)
    return df.sort_values(by='fold', ascending=True).reset_index(drop=True) # sort by fold