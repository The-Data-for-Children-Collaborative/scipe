import numpy as np
import pandas as pd
import csv
from tifffile import imread
from copy import deepcopy
from osgeo import gdal

raster_path = './100m/'
survey_path = './Survey/'

def write_raster(arr,file_match,file_out): # writes arr to raster file_out with geospatial coordinates of file_match
    # load match dataset
    ds_in = gdal.Open(file_match)
    cols, rows = arr.shape[0:2]
    # write data to matching raster
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(file_out, rows, cols, 1, gdal.GDT_Byte)
    ds_out.SetGeoTransform(ds_in.GetGeoTransform())
    ds_out.SetProjection(ds_in.GetProjection())
    ds_out.GetRasterBand(1).WriteArray(arr[:,:,0])
    ds_out.GetRasterBand(1).SetNoDataValue(0)
    ds_out.FlushCache() # save to disk

def get_feature_names():
    feature_names = [f'landsat_b{i}' for i in range(10)]
    feature_names += ['ndvi', 'ndwi','ntl','hrsl','road_dist_m','building_area_diss','building_area_spacenet','building_area_spacesur',
             'no_class','closed_forest','open_forest','shrubs','hb_veg',
             'hb_waste','moss','bare','cropland','urban','snow','water','sea']
    return feature_names

def load_rasters(roi): # load master list of all feature rasters
    rasters = []
    rasters.append(imread(f'{raster_path}{roi}_landsat_100m.tif').astype('float32'))
    rasters.append(imread(f'{raster_path}{roi}_ndvi_100m.tif').astype('float32'))
    rasters.append(imread(f'{raster_path}{roi}_ndwi_100m.tif').astype('float32'))
    rasters.append(imread(f'{raster_path}{roi}_ntl_100m.tif').astype('float32'))
    rasters.append(imread(f'{raster_path}{roi}_hrsl_100m.tif').astype('uint8'))
    rasters.append(imread(f'{raster_path}{roi}_roads_dist_100m.tif').astype('float32'))
    rasters.append(imread(f'{raster_path}{roi}_building_area_100m.tif').astype('float32'))
    rasters.append(imread(f'{raster_path}{roi}_building_area_spacenet_100m.tif').astype('float32'))
    rasters.append(imread(f'{raster_path}{roi}_building_area_spacesur_100m.tif').astype('float32'))
    rasters.append(imread(f'{raster_path}{roi}_landcover_100m.tif').astype('uint8'))
    return rasters

def get_context(raster,y,x): # return mean of 3x3 context area surrounding point (x,y) of raster
    def in_bounds(raster,y,x):
        return 0 <= y < raster.shape[0] and 0 <= x < raster.shape[1]
    vals = [raster[i,j] for i in [y-1,y,y+1] for j in [x-1,x,x+1] if in_bounds(raster,i,j)]
    return np.mean(np.array(vals), axis=0)

def construct_dataset(feature_names,rasters,pop,context=True): # construct dataframe from rasters, survey
    d = {'x':[],'y':[]} # coordinates used to perform validation split spatially
    for f in feature_names:
        d[f] = []
    if context:
        for f in feature_names:
            d[f'{f}_context'] = []
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
                if context: # consider context around cell
                    for r in rasters:
                        row = np.append(row,get_context(r,i,j))
                row = np.append(row,n)
                df.loc[count] = row
                count+=1
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)
    return df

# splits the dataset spatially into (n x n) segments with approx equal numbers of survey points by:
# 1. splitting into n segments by y coordinate, then
# 2. splitting each segment into n segments by x coordinate
def get_val_split(df,n=2,coord='y',leaf=False):
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

# returns a dataframe containing a concatenation of the dataframes in dfs each labelled with a different value in appended 'folds' column
def label_folds(dfs):
    df_combined = pd.DataFrame()
    for i,df in enumerate(dfs):
        df['fold'] = i
        df_combined = df_combined.append(df,ignore_index=True)
    return df_combined

# Return dataframe with outliers defined in csv file removed
def remove_outliers(df,outliers,strong=True,weak=True):
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
                