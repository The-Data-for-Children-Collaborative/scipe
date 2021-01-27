from osgeo import gdal, gdalconst
from tifffile import imread, imsave
from shutil import copyfile
import numpy as np
import json
import os

from functions.utilies import project_raster, proximity_raster

resamplers_table = {'average': gdalconst.GRA_Average, 'nearest': gdalconst.GRA_NearestNeighbour, 'max': gdalconst.GRA_Max}

'''
    For each roi, reproject input rasters to the relevant reference grid, using
    dictated resampling method.
    
            Parameters:
                    in_dir (str): The path to the input data directory
                    in_rasters (str): The names of input rasters within in_dir
                    ref_rasters (list): The path to reference grid for each roi
                    out_dir (str): The output directory
                    rois (list): The names of rois
                    resampling (list): The resampling technique used for each input raster

            Returns:
                    None
'''
def reproject_data(in_dir,in_rasters,ref_rasters,out_dir,rois,resampling):
    for i,roi in rois:
        ref_path = ref_rasters[i]
        for j,raster in enumerate(in_rasters):
            in_path = os.path.join(in_dir,raster)
            out_path = os.path.join(*[out_dir,roi,raster])
            project_raster(in_path,ref_path,out_path,resampling[j])

            
'''
    Preprocess data according to specifications in file params_path. All
    output is to disk.
    
            Parameters:
                    params_path (str): A JSON file holding parameters

            Returns:
                    None
'''
def preprocess_data(params_path):
    
    # load and parse params
    params = {}
    with open(params_path,'r') as f:
        params = json.load(f)                        
    rois = params['rois']
    in_dir = params['input_dir']
    in_rasters = params['input_rasters']
    resampling = [resamplers_table[r] for r in params['resampling']]
    out_dir = params['output_dir']
    pop_rasters = params['pop_rasters'] # one for each roi
    
    # rasterize vector data 
    
    # reproject data to reference grid                   
    reproject_data(in_dir,in_rasters,pop_rasters,out_dir,rois,resampling)
    
    # compute indices and simplify landcover


def building_area(roi,building_path,cat='spacesur'):
    shape = imread(f'./data/pop/{roi}_pop.tif').shape
    area_raster = np.zeros(shape)
    buildings_dir = f'{building_path}{roi}/'
    for file in os.listdir(buildings_dir):
        if file.endswith('.tif'):
            buildings = imread(os.path.join(buildings_dir,file))
            y,x = file.split('.')[0].split('_')
            y,x = int(y),int(x)
            building_area = np.sum(buildings) * 0.25 # each pixel occupies 0.25 m^2
            area_raster[y,x] = building_area
    imsave(f'./data/100m/{roi}_building_area_{cat}_100m.tif',area_raster)
    
def conv_class(raster): # lookup table for land cover classification simplification
    # 0 = no data, 1 = closed forest, 2 # open forest, 3 = shrubs, 4 = herbaceous vegetation, 
    # 5 = herbaceous wasteland, 6 = moss and lichen, 7 = bare/sparse vegetation, 8 = cropland,
    # 9 = urban/built up, 10 = snow and ice, 11 = permanent water body, 12 = open sea
    d = {'0': 0, '111': 1, '112': 1, '113': 1, '114': 1, '115': 1, '116': 1, '121': 2,
         '122': 2, '123': 2, '124': 2, '125': 2, '126': 2, '20': 3, '30': 4, '90': 5,
         '100': 6, '60': 7, '40': 8, '50': 9, '70': 10, '80': 11, '200': 12}
    # convert raster
    for i in range(raster.shape[0]):
        for j in range(raster.shape[1]):
            raster[i,j] = d[str(raster[i,j])]
    return raster

def ndvi_landsat(file_in,file_out):
    landsat = imread(file_in)
    nir = landsat[:,:,4]
    red = landsat[:,:,3]
    ndvi = (nir - red) / (nir + red)
    imsave(file_out,ndvi)
    
def ndwi_landsat(file_in,file_out):
    landsat = imread(file_in)
    nir = landsat[:,:,4]
    swir = landsat[:,:,5]
    ndwi = (nir - swir) / (nir + swir)
    imsave(file_out,ndwi)

def process_district(roi,model_paths,model_names):
    match_filename=f'./data/pop/{roi}_pop.tif'

    src_filename=f'./data/landsat/{roi}_landsat_2019.tif'
    dst_filename=f'./data/100m/{roi}_landsat_100m.tif'
    project_raster(src_filename,match_filename,dst_filename,gdalconst.GRA_Average,10)
    
    ndvi_landsat(dst_filename,f'./data/100m/{roi}_ndvi_100m.tif')
    ndwi_landsat(dst_filename,f'./data/100m/{roi}_ndwi_100m.tif')

    src_filename=f'./data/ntl/{roi}_ntl_20190401.tif'
    dst_filename=f'./data/100m/{roi}_ntl_100m.tif'
    project_raster(src_filename,match_filename,dst_filename,gdalconst.GRA_Average,1)
    
    src_filename=f'./data/roads/{roi}_roads.tif'
    dst_filename=f'./data/100m/{roi}_roads_100m.tif'
    project_raster(src_filename,match_filename,dst_filename,gdalconst.GRA_NearestNeighbour,1)
    
    src_filename=f'./data/hrsl/hrsl.tif'
    dst_filename=f'./data/100m/{roi}_hrsl_100m.tif'
    project_raster(src_filename,match_filename,dst_filename,gdalconst.GRA_Max,1)
    
    src_filename=f'./data/land_cover/landcover_2019.tif'
    dst_filename=f'./data/100m/{roi}_landcover_100m.tif'
    project_raster(src_filename,match_filename,dst_filename,gdalconst.GRA_NearestNeighbour,1)
    landcover = imread(dst_filename).astype('uint8')
    converted = conv_class(landcover)
    # one hot encode landcover
    converted_enc = np.zeros((converted.shape[0],converted.shape[1],13)) 
    for i in range(converted.shape[0]): 
        for j in range(converted.shape[1]):
            v = converted[i,j]
            converted_enc[i,j,v] = 1
    converted = converted_enc
    imsave(dst_filename,converted)
    
    proximity_raster(f'./data/100m/{roi}_roads_100m.tif',f'./data/100m/{roi}_roads_dist_100m.tif')
    
    for i,path in enumerate(model_paths):    
        building_area(roi,f'{path}pred/',cat=model_names[i])