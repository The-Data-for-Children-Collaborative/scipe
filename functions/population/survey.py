"""Module for rasterizing survey data. NOTE: this is dependent on survey details, so must be modified for each survey
format. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from osgeo import gdal, ogr


def get_extent(df):
    """ Return extent of survey cells contained within df. """
    xs = df['cell_x'].to_numpy()
    ys = df['cell_y'].to_numpy()
    # remove NaN
    xs = xs[~np.isnan(xs)]
    ys = ys[~np.isnan(ys)]
    # find min and max
    min_x = int(np.min(xs))
    max_x = int(np.max(xs))
    min_y = int(np.min(ys))
    max_y = int(np.max(ys))

    return min_x, max_x, min_y, max_y


def get_arr(df, extent, feature, verbose=False):
    """ Return numpy array of target feature from survey dataframe. """
    min_x, max_x, min_y, max_y = extent
    # Initialize population array using extent
    arr = np.zeros(((max_y - min_y) // 100 + 1, (max_x - min_x) // 100 + 1))
    nan_count = 0
    # Iterate through dataframe adding population to array
    for index, row in df.iterrows():
        x = row['cell_x']
        y = row['cell_y']
        if math.isnan(x) or math.isnan(y):  # filter out positionless values
            nan_count += 1
            continue
        x = (int(x) - min_x) // 100
        y = (max_y - int(y)) // 100
        if feature == 'count_buildings':
            arr[y, x] += 1
            continue
        arr[y, x] += row[feature]
    if verbose:
        print(f'Ignored {nan_count} values without position defined')
    return np.where(arr > 0, arr, np.nan)  # set zero population grid elements to nan so they save to image properly


def arr_to_raster(out_file, origin, pixel_width, pixel_height, srs, array):
    """ Save survey array to raster. """
    ds = ogr.Open(srs)
    layer = ds.GetLayer()
    srs = layer.GetSpatialRef()

    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(out_file, cols, rows, 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform((origin[0], pixel_width, 0, origin[1], 0, -pixel_height))
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(0)

    out_raster.SetProjection(srs.ExportToWkt())
    out_band.FlushCache()


def df_to_raster(df, filename, srs_path, feature):
    """ Rasterize survey and save to disk. """
    # Calculate extent of sample grid
    extent = get_extent(df)
    min_x, _, min_y, max_y = extent
    # Calculate population array from samples
    pop = get_arr(df, extent, feature)
    # Export population data as geotiff
    arr_to_raster(filename, (min_x - 50, max_y + 50), 100, 100, srs_path, pop)


def display_surveys(dfs):
    """ Plot survey rasters. """
    f, axarr = plt.subplots(1, len(dfs), figsize=(17.5, 10))
    for i, df in enumerate(dfs):
        pop = get_arr(df, get_extent(df), 'members_n')
        axarr[i].imshow(np.where(pop != np.nan, 1, 0))


def rasterize_survey(params):
    """ Process population survey for each roi according to params. """
    rois = params['rois']
    survey_paths = params['survey_paths']
    srs_paths = params['srs_paths']
    out_path = params['out_dir']

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for (roi, survey_path, srs_path) in zip(rois, survey_paths, srs_paths):
        print(f'Rasterizing {roi} survey... ', end='')
        df = pd.read_stata(survey_path)
        df_to_raster(df, os.path.join(out_path, f'{roi}_pop.tif'), srs_path, 'members_n')
        print('done.')
