"""
Module for preprocessing remote sensing data, including reprojecting data to match microcensus grid.

Todo:
    * Take vector road data as input and rasterize in pipeline.
    * Compute road density vs. distance to road.

"""
from osgeo import gdalconst
from tifffile import imread, imsave
from tqdm import tqdm
import numpy as np
import os

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utilities import project_raster, proximity_raster, write_raster

lcc_class_values = {0: 0, 111: 1, 112: 1, 113: 1, 114: 1, 115: 1, 116: 1, 121: 2,
                    122: 2, 123: 2, 124: 2, 125: 2, 126: 2, 20: 3, 30: 4, 90: 5,
                    100: 6, 60: 7, 40: 8, 50: 9, 70: 10, 80: 11, 200: 12}
"""dict: Maps raw Coepernicus LCC class values to simplified class values. """

lcc_class_names = {0: 'no_data', 1: 'closed_forest', 2: 'open_forest', 3: 'shrubs', 4: 'hb_veg', 5: 'hb_waste',
                   6: 'moss', 7: 'sparse', 8: 'cropland', 9: 'urban', 10: 'snow', 11: 'water', 12: 'sea'}
"""dict: Maps simplified class values to names. """

resamplers_table = {'average': gdalconst.GRA_Average, 'nearest': gdalconst.GRA_NearestNeighbour,
                    'max': gdalconst.GRA_Max}
"""dict: Maps resmpler name to GDAL resampler code. """


def ndvi_landsat_f(raster):
    """ Calculate ndvi for landsat raster. """
    nir = raster[:, :, 4]
    red = raster[:, :, 3]
    ndvi = (nir - red) / (nir + red)
    return ndvi


def ndwi_landsat_f(raster):
    """ Calculate ndwi for landsat raster. """
    nir = raster[:, :, 4]
    swir = raster[:, :, 5]
    ndwi = (nir - swir) / (nir + swir)
    return ndwi


indices_table = {'ndvi': ndvi_landsat_f, 'ndwi': ndwi_landsat_f}
"""dict: Maps index name to function that calculates it on Landsat data. """


def reproject_data(in_dir, in_rasters, ref_rasters, out_dir, rois, resampling):
    """
    Reproject input rasters to the relevant reference grid, using dictated resampling method.

    Args:
            in_dir (str): The path to the input data directory.
            in_rasters (str): The names of input rasters within in_dir.
            ref_rasters (:obj:`list` of :obj:`str`): The path to reference grid for each roi.
            out_dir (str): The output directory.
            rois (:obj:`list` of :obj:`str`): The names of rois.
            resampling (:obj:`list` of :obj:`int`): The resampling technique code used for each input raster.

    Returns:
            None.
    """
    for i, roi in enumerate(rois):
        if not os.path.exists(os.path.join(out_dir, roi)):
            os.makedirs(os.path.join(out_dir, roi))
        ref_path = ref_rasters[i]
        for j, raster in enumerate(in_rasters):
            in_path = os.path.join(in_dir, raster)
            out_path = os.path.join(*[out_dir, roi, raster])
            project_raster(in_path, ref_path, out_path, resampling[j])


def compute_indices(out_dir, filename, rois, indices):
    """
    Calculate listed indices for each roi using landsat data from filename.

    Args:
            out_dir (str): The out path for processed data, including indices.
            filename (str): The name of the landsat data in each roi's directory.
            rois (:obj:`list` of :obj:`str`): The names of rois.
            indices (:obj:`list` of :obj:`str`): The names of indices to be computed (ndvi and ndwi supported).

    Returns:
            None.
    """
    index_functions = [indices_table[index] for index in indices]
    for roi in rois:
        raster = imread(os.path.join(*[out_dir, roi, filename]))
        for f, index in zip(index_functions, indices):
            index_raster = f(raster)
            imsave(os.path.join(*[out_dir, roi, (index + '.tif')]), index_raster)


def compute_road_dist(out_dir, filename, rois):
    """
    Calculate distance to road for each roi using road data from filename.

    Args:
            out_dir (str): The out path for processed data.
            filename (str): The name of the road data in each roi's directory.
            rois (:obj:`list` of :obj:`str`): The names of rois.

    Returns:
            None.
    """
    for roi in rois:
        file_in = os.path.join(*[out_dir, roi, filename])
        file_out = os.path.join(*[out_dir, roi, filename.replace('.', '_dist.')])
        proximity_raster(file_in, file_out)
        os.remove(file_in)  # clean up


def process_lcc(src_dir, filename, rois):
    """
    Process landcover classification by simplifying classes and saving to per-class rasters.

    Args:
            src_dir (str): The path to the directory containing lcc for each roi.
            filename (str): The name of the lcc in each roi's directory.
            rois (:obj:`list` of :obj:`str`): The names of rois.

    Returns:
            None.
    """
    for roi in rois:
        file_in = os.path.join(*[src_dir, roi, filename])
        raster = imread(file_in).astype('uint8')
        n_classes = lcc_class_values[max(lcc_class_values, key=lcc_class_values.get)] + 1
        raster_onehot = np.zeros((raster.shape[0], raster.shape[1], n_classes))
        # simplify classification and one hot encode
        for i in range(raster.shape[0]):
            for j in range(raster.shape[1]):
                v = lcc_class_values[raster[i, j]]
                raster_onehot[i, j, v] = 1
        for c in range(n_classes):  # save each class to raster
            file_out = os.path.join(*[src_dir, roi, (lcc_class_names[c] + '.tif')])
            imsave(file_out, raster_onehot[:, :, c])
        os.remove(file_in)  # clean up


def process_landsat(src_dir, filename, rois):  # land cover classification simplification
    """
    Split landsat data by saving to per-band rasters.

    Args:
            src_dir (str): The path to the directory containing landsat for each roi.
            filename (str): The name of the landsat raster in each roi's directory.
            rois (list): The names of rois.

    Returns:
            None.
    """
    for roi in rois:
        file_in = os.path.join(*[src_dir, roi, filename])
        raster = imread(file_in)
        n_bands = raster.shape[2]
        for b in range(n_bands):
            file_out = os.path.join(*[src_dir, roi, filename.replace('.', f'_b{b}.')])
            imsave(file_out, raster[:, :, b])
        os.remove(file_in)  # clean up


def process_footprints(footprint_dirs, model_names, thresholds, pop_rasters, rois, out_dir):
    """
    Compute building footprint area for each roi for all tiles in footprints_dir.

    Args:
            footprint_dirs (:obj:`list` of :obj:`str`): The path to the footprint directory for each model.
            model_names (:obj:`list` of :obj:`str`): The name of each model.
            thresholds (:obj:`list` of :obj:`float`): The threshold used for each model.
            pop_rasters (:obj:`list` of :obj:`str`): The path to reference grid for each roi.
            rois (:obj:`list` of :obj:`str`): The names of rois.
            out_dir (str): The output directory.

    Returns:
            None
    """
    for footprint_dir, model_name, threshold in zip(footprint_dirs, model_names, thresholds):
        for roi, pop_raster in zip(rois, pop_rasters):
            shape = imread(pop_raster).shape
            area_raster = np.zeros(shape)
            roi_dir = os.path.join(footprint_dir, roi)
            for file in tqdm(os.listdir(roi_dir)):
                if file.endswith('.tif'):
                    buildings = imread(os.path.join(roi_dir, file))
                    buildings = np.where(buildings > threshold, 0.25, 0)  # each pixel occupies 0.25 m^2
                    y, x = file.split('.')[0].split('_')
                    y, x = int(y), int(x)
                    building_area = np.sum(buildings)
                    area_raster[y, x] = building_area
            write_raster(area_raster, pop_raster, os.path.join(out_dir, roi, f'building_area_{model_name}.tif'))


def preprocess_data(params):
    """
    Preprocess data according to specifications in file params_path. All rasters output to disk.

    Args:
            params (dict): A dictionary holding parameters.

    Returns:
            None.
    """
    # parse params
    in_dir = params['input_dir']
    in_rasters = params['input_rasters']
    resampling = [resamplers_table[r] for r in params['resampling']]
    out_dir = params['output_dir']
    rois = params['rois']
    pop_rasters = params['pop_rasters']
    landcover = params['landcover']
    landsat = params['landsat']
    roads = params['roads']
    footprint_dirs = params['footprint_dirs']
    model_names = params['model_names']
    thresholds = params['thresholds']
    indices = params['indices']

    # initialize directories
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # TODO: rasterize vector data

    # reproject data to reference grid
    print("Reprojecting data... ", end="")
    reproject_data(in_dir, in_rasters, pop_rasters, out_dir, rois, resampling)
    print("done")

    # compute indices
    print("Computing indices... ", end="")
    compute_indices(out_dir, landsat, rois, indices)
    print("done")

    # compute distance to road
    print("Computing distance to road... ", end="")
    compute_indices(out_dir, landsat, rois, indices)
    print("done")

    # process landcover and landsat data
    print("Splitting data... ", end="")
    process_landsat(out_dir, landsat, rois)
    process_lcc(out_dir, landcover, rois)
    print("done")

    # compute building footprint area
    print("Calculating footprint area... ")
    process_footprints(footprint_dirs, model_names, thresholds, pop_rasters, rois, out_dir)
    print("done")
