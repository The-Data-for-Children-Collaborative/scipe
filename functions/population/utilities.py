from osgeo import gdal, gdalconst
import numpy as np


def write_raster(arr, match_filename, dst_filename, dtype=gdal.GDT_Float32):
    """
    Write array to disk with georeferencing of match file.

    Args:
        arr (np.ndarray): Array to write to disk.
        match_filename (str): geoTiff to match georeferencing of output with.
        dst_filename (str): Path to write array to.
        dtype (int): Gdal datatype code to write with.

    Returns:
        None
    """
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)  # grayscale image
    # load match dataset
    ds_in = gdal.Open(match_filename)
    cols, rows = arr.shape[0:2]
    # write data to matching raster
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(dst_filename, rows, cols, 1, dtype)
    ds_out.SetGeoTransform(ds_in.GetGeoTransform())
    ds_out.SetProjection(ds_in.GetProjection())
    ds_out.GetRasterBand(1).WriteArray(arr[:, :, 0])
    ds_out.GetRasterBand(1).SetNoDataValue(0)
    ds_out.FlushCache()  # save to disk


def project_raster(src_filename, match_filename, dst_filename, resampling, n_bands=0):
    """
    Reproject source raster to match georeferencing of match raster, using specified resampling technique. Outputs to disk.

    Args:
        src_filename (str): Path to raster to reproject.
        match_filename (str): geoTiff to reproject to.
        dst_filename (str): Path to write reprojected raster to.
        resampling (int): GDAL resmapling code.
        n_bands (:obj:`int`, optional): Number of bands to reproject. Defaults to 0 (all bands).

    Returns:
        None
    """

    # source raster
    src_ds = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src_ds.GetProjection()

    # raster to match
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()

    if n_bands < 1:
        n_bands = src_ds.RasterCount
    if n_bands > 10:  # bug with GDAL, can only use first 10 bands
        n_bands = 10

    # output/destination
    dst = gdal.GetDriverByName('Gtiff').Create(dst_filename, match_ds.RasterXSize, match_ds.RasterYSize, n_bands,
                                               gdalconst.GDT_Float32)
    dst.GetRasterBand(1).SetNoDataValue(0)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    gdal.ReprojectImage(src_ds, dst, src_proj, match_proj, resampling)

    del dst  # flush to save to disk TODO: shouldn't be required


def proximity_raster(src_filename, dst_filename):
    """
    Calculate Euclidean distance to closest True cell for boolean src raster. Outputs to disk.

    Args:
        src_filename (str): Path to raster to run distance calculation on.
        dst_filename (str): Path to write distance raster to.

    Returns:
        None
    """

    # source
    src_ds = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src_ds.GetProjection()
    src_geotrans = src_ds.GetGeoTransform()

    # output/destination
    dst = gdal.GetDriverByName('Gtiff').Create(dst_filename, src_ds.RasterXSize, src_ds.RasterYSize, 1,
                                               gdalconst.GDT_Float32)
    dst.GetRasterBand(1).SetNoDataValue(-1)
    dst.SetGeoTransform(src_geotrans)
    dst.SetProjection(src_proj)

    gdal.ComputeProximity(src_ds.GetRasterBand(1), dst.GetRasterBand(1), ["DISTUNITS=GEO"])

    del dst  # flush to save to disk TODO: shouldn't be required
