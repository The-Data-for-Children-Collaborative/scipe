from osgeo import gdal, gdalconst
from tifffile import imread, imsave
from shutil import copyfile
import numpy as np
import os

def project_raster(src_filename,match_filename,dst_filename,resampling,n_bands):
    #source
    src_ds = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src_ds.GetProjection()
    src_geotrans = src_ds.GetGeoTransform()

    #raster to match
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()

    #output/destination
    dst = gdal.GetDriverByName('Gtiff').Create(dst_filename, match_ds.RasterXSize, match_ds.RasterYSize, n_bands, gdalconst.GDT_Float32)
    dst.GetRasterBand(1).SetNoDataValue(0)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    gdal.ReprojectImage(src_ds, dst, src_proj, match_proj, resampling)

    del dst # flush to save to disk
    
def proximity_raster(src_filename,dst_filename):
    #source
    src_ds = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src_ds.GetProjection()
    src_geotrans = src_ds.GetGeoTransform()

    #output/destination
    dst = gdal.GetDriverByName('Gtiff').Create(dst_filename, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdalconst.GDT_Float32)
    dst.GetRasterBand(1).SetNoDataValue(-1)
    dst.SetGeoTransform(src_geotrans)
    dst.SetProjection(src_proj)

    gdal.ComputeProximity(src_ds.GetRasterBand(1), dst.GetRasterBand(1), ["DISTUNITS=GEO"])

    del dst # flush to save to disk
    
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
    d = {}
    d['0'] = 0 # no data
    d['111'] = 1 # closed forest
    d['112'] = 1 # closed forest
    d['113'] = 1 # closed forest
    d['114'] = 1 # closed forest
    d['115'] = 1 # closed forest
    d['116'] = 1 # closed forest
    d['121'] = 2 # open forest
    d['122'] = 2 # open forest
    d['123'] = 2 # open forest
    d['124'] = 2 # open forest
    d['125'] = 2 # open forest
    d['126'] = 2 # open forest
    d['20'] = 3 # shrubs
    d['30'] = 4 # herbaceous vegetation
    d['90'] = 5 # herbaceous wasteland
    d['100'] = 6 # moss and lichen
    d['60'] = 7 # bare/sparse vegetation
    d['40'] = 8 # cropland
    d['50'] = 9 # urban/built up
    d['70'] = 10 # snow and ice
    d['80'] = 11 # permanent water body
    d['200'] = 12 # open sea
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