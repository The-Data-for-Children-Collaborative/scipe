""" Script for splitting imagery on survey grid.

Usage:
    ``python split_imagery.py <path_to_imagery> <path_to_survey> <out_path>``
 """

import os
import subprocess
import shutil
import sys

from tqdm import tqdm
from osgeo import gdal, gdalconst, osr


def build_vrt(files_in, file_out, src_extent, target_extent):
    """
    Wrapper for gdalbuildvrt, used to build virtual raster dataset.

    Args:
        files_in: List of files to build into vrt.
        file_out: Name of output vrt.
        src_extent: Extent to build vrt on from files_in.
        target_extent: Extent to warp vrt to after building.

    Returns:
        None

    """
    out_tmp = file_out.replace('.vrt', '_src.vrt')
    cmd = f'gdalbuildvrt -te {src_extent[0]} {src_extent[1]} {src_extent[2]} {src_extent[3]} {out_tmp}'
    for f in files_in:
        cmd += " " + f
    print("Building virtual dataset")
    subprocess.call(cmd)
    height, width = get_shape(out_tmp)
    x_min, y_min, x_max, y_max = target_extent
    cmd = f'gdalwarp -overwrite -s_srs EPSG:4326 -t_srs EPSG:32736 -ts {width} {height} -te {x_min} {y_min} {x_max} {y_max} {out_tmp} {file_out}'
    subprocess.call(cmd)


def crop_img(img, out_dir, grid_shape, x_offset, y_offset, img_ext):
    """
    Crop large image into several smaller ones.

    Args:
        img: Path to image to be cropped (can be virtual raster).
        out_dir: Directory to output crops to.
        grid_shape: Shape of grid we are cropping over.
        x_offset: Width of crop in georeferenced units.
        y_offset: Heihgt of crop in georeferenced units.
        img_ext: Spatial extent of image to be cropped, in georeferenced units.

    Returns:
        None
    """
    base_cmd = 'gdal_translate -q -of GTIFF -ot BYTE -co COMPRESS=JPEG -co JPEG_QUALITY=90 -projwin'
    print("Slicing dataset")
    with tqdm(total=grid_shape[1] * grid_shape[0]) as pbar:
        for y in range(grid_shape[0]):
            for x in range(grid_shape[1]):
                src_win = f'{img_ext[0] + x * x_offset} {img_ext[3] - y * y_offset} {img_ext[0] + (x + 1) * x_offset} {img_ext[3] - (y + 1) * y_offset}'
                out_name = f'{out_dir}{y}_{x}.tif'
                cmd = f'{base_cmd} {src_win} {img} {out_name}'
                os.system(cmd)
                pbar.update(1)


def get_srs(img):
    """ Returns spatial referece of geoTiff img. """
    data = gdal.Open(img, gdalconst.GA_ReadOnly)
    return osr.SpatialReference(wkt=data.GetProjection())


def get_extent(img, epsg=None):
    """ Return extent of geoTiff img, in SRS associated with epsg if specified. """
    if epsg:
        new_img = './tmp/proj.tif'
        gdal.Warp(new_img, img, dstSRS=f'EPSG:{epsg}')
        img = new_img
    ds = gdal.Open(img, gdalconst.GA_ReadOnly)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]
    #     if epsg:
    #         os.remove(img) # remove temporary reprojected image
    return minx, miny, maxx, maxy


def get_shape(img):
    ds = gdal.Open(img, gdalconst.GA_ReadOnly)
    return ds.RasterYSize, ds.RasterXSize


def in_range(n, bounds):
    return bounds[0] <= n <= bounds[1]


def overlapping(img_ext, ref_ext):
    """ Returns true if extent img_ext overlaps with extent ref_ext. """
    x_bounds = (ref_ext[0], ref_ext[2])
    y_bounds = (ref_ext[1], ref_ext[3])
    x_in_range = in_range(img_ext[0], x_bounds) or in_range(img_ext[2], x_bounds)
    y_in_range = in_range(img_ext[1], y_bounds) or in_range(img_ext[3], y_bounds)
    return x_in_range and y_in_range


#
def get_overlap_imgs(src_dir, ref_ext):
    """ Returns list of geoTiffs in src_dir that are within the extent of ref_img. """
    imgs = []
    for file in os.listdir(src_dir):
        if file.endswith(".tif"):
            file_path = os.path.join(src_dir, file)
            img_ext = get_extent(file_path)
            if overlapping(img_ext, ref_ext):  # image overlaps with reference area
                imgs.append(file_path)
    return imgs


def get_all(directory, ext):
    """ Get all files with extension ext in directory. """
    fs = []
    for file in os.listdir(directory):
        if file.endswith(ext):
            fs.append(os.path.join(directory, file))
    return fs


def remove_all(directory, ext):
    """ Removes all files with extension ext in directory. """
    print(f"Removing files in {directory} with extension {ext}")
    for file in tqdm(os.listdir(directory)):
        if file.endswith(ext):
            os.remove(os.path.join(directory, file))


#
def split_dataset(src_dir, ref_img, out_dir, tmp_dir='./tmp/'):
    """ Split geoTiffs in src_dir into image patches covering each pixel of ref_img. """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)  # for gdal temporary files

    src_examples = [f for f in os.listdir(src_dir) if f.endswith('.tif')]  # get valid files from src_dir
    if len(src_examples) > 0:
        src_example = os.path.join(src_dir,src_examples[0])
    else:
        print("No geoTiffs found in src_dir.")
        return
    src_epsg = get_srs(src_example).GetAttrValue('AUTHORITY', 1)  # get source epsg
    ref_ext_reproj = get_extent(ref_img, epsg=src_epsg)  # get extent of ref img in source spatial reference
    img_paths = get_overlap_imgs(src_dir, ref_ext_reproj)
    vrt_path = os.path.join(out_dir, 'mosaic.vrt')
    build_vrt(img_paths, vrt_path, ref_ext_reproj, get_extent(ref_img))

    ref_shape = get_shape(ref_img)
    img_ext = get_extent(vrt_path)
    x_len, y_len = abs(img_ext[2] - img_ext[0]), abs(img_ext[3] - img_ext[1])
    x_offset, y_offset = x_len / ref_shape[1], y_len / ref_shape[0]

    crop_img(vrt_path, out_dir, ref_shape, x_offset, y_offset, img_ext)
    remove_all(out_dir, '.msk')
    shutil.rmtree(tmp_dir)  # remove temporary files


if __name__ == '__main__':
    split_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
