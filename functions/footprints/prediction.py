import tensorflow as tf
import numpy as np
import json
import os
from tifffile import imsave
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import smart_resize, load_img, img_to_array

from population.data import in_bounds
from population.utilities import write_raster

def load_model(model_path, weights_path):
    """ Load Keras model and weights from disk. """
    with open(model_path) as json_file:
        model_json = json.dumps(json.load(json_file))
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights_path)
        return model


def load_models(model_paths):
    """ Load list of Keras models and weights from disk. """
    return [load_model(path + 'model.json', path + 'model.h5') for path in model_paths]


def learn_distribution(path, count):
    """ Learn distribution of imagery by sampling subset. """
    ls = np.array(os.listdir(path))
    n = ls.shape[0]
    idxs = np.random.randint(0, n, (count,))
    mean = np.zeros(3, )
    std = np.zeros(3, )
    print(f'Learning channel-wise distribution for imagery in {path}', flush=True)
    for i in tqdm(idxs):
        file = f'{path}{ls[i]}'
        if file.endswith('.tif'):
            img = img_to_array(load_img(file)) / 255
            mean += np.mean(img, axis=(0, 1)) / count
            std += np.std(img, axis=(0, 1)) / count
    return mean, std


def predict_sample(img, mean, std, model):
    """
    Standardize image with featurewise mean and standard deviation then make prediction with model.

    Args:
            img (:obj:`np.ndarray`): Input image to model.
            mean (:obj:`np.ndarray`): The feature-wise mean used to center img.
            std (:obj:`np.ndarray`): The feature-wise standard deviation used to scale img.
            model (:obj:`tf.keras.Model`): Keras model used for predictions.

    Returns:
            :obj:`np.ndarray`: Model prediction.
    """
    original_size = img.shape[0:2]
    img = tf.image.resize(img, model.input_shape[1:3])  # resize to match model input size
    img = ((img / 255) - mean) / std
    img = np.expand_dims(img, axis=0)
    out = model.predict(img)[0]
    out = tf.image.resize(out, original_size)  # resize back to original size
    return out


def estimate_footprints(roi, survey, img_dir, model_dirs, context_sizes, n_samples=1000):
    """
    Estimate footprints using specified models for roi across survey tiles and surrounding context area, and save to disk.

    Args:
            roi (str): The roi to predict over, used for writing to disk.
            survey (:obj:`np.ndarray`): The survey whose tiles to predict on.
            img_dir (str): The path to the directory containing tiles.
            model_dirs (:obj:`list` of :obj:`str`): The list of paths to models used for prediction.
            context_sizes (:obj:`list` of :obj:`int`): The list of context sizes to predict on around survey tiles.
            n_samples (:obj:`int`, optional): The number of samples to use to sample featurewise mean and standard deviation.

    Returns:
            None.
    """
    context_size = max(context_sizes)
    mean, std = learn_distribution(img_dir, n_samples)
    print(mean, std)
    models = load_models(model_dirs)
    visited = set()

    for model_dir in model_dirs:
        path = os.path.join(*[model_dir, 'pred', roi])
        if not os.path.exists(path):
            os.makedirs(path)

    print(f'Predicting building footprints for survey tiles in {img_dir}', flush=True)
    with tqdm(total=np.count_nonzero(survey)) as pbar:
        for y in tqdm(range(survey.shape[0])):
            for x in range(survey.shape[1]):
                if survey[y, x] > 0:  # only predict for survey locations and surrounding contexts
                    # predict over both x and y of context
                    for x_inc in range((1 - context_size) // 2, (1 + context_size) // 2):
                        for y_inc in range((1 - context_size) // 2, (1 + context_size) // 2):
                            tile_x, tile_y = x + x_inc, y + y_inc
                            if (tile_y, tile_x) not in visited and in_bounds(survey, tile_y, tile_x):
                                img_path = f'{img_dir}{tile_y}_{tile_x}.tif'
                                img = img_to_array(load_img(img_path))
                                for i, model in enumerate(models):
                                    dst = f'{model_dirs[i]}pred/{roi}/{tile_y}_{tile_x}.tif'
                                    out = np.array(predict_sample(img, mean, std, model)).astype(
                                        'float32')  # TODO: speed up by performing in batches
                                    write_raster(out, img_path, dst)
                                visited.add((tile_y, tile_x))  # avoid duplicating work due to overlapping contexts
                                pbar.update(1)


def estimate_footprints_full(roi, survey, img_dir, model_dirs, context_sizes, n_samples=1000):
    """
    Estimate footprints for entire roi using specified model, and save to disk.

    Args:
            roi (str): The roi to predict over, used for writing to disk.
            survey (:obj:`np.ndarray`): The survey whose bounds to predict over.
            img_dir (str): The path to the directory containing tiles.
            model_dirs (str): Path to the model used for prediction.
            context_sizes (:obj:`list` of :obj:`int`): The list of context sizes to predict on around survey tiles.
            n_samples (:obj:`int`, optional): The number of samples to use to sample featurewise mean and standard deviation.

    Returns:
            None.
    """
    context_size = max(context_sizes)
    mean, std = learn_distribution(img_dir, n_samples)
    print(mean, std)
    models = load_models(model_dirs)

    for model_dir in model_dirs:
        path = os.path.join(*[model_dir, 'pred', roi])
        if not os.path.exists(path):
            os.makedirs(path)

    print(survey.shape)

    for i, model in enumerate(models):
        print(f'Predicting building footprints for all tiles in {img_dir}', flush=True)
        for y in tqdm(range(survey.shape[0])):  # process data in vertical strips
            imgs = [(((img_to_array(load_img(f'{img_dir}{y}_{x}.tif')) / 255) - mean) / std) for x in
                    range(survey.shape[1])]
            shapes = [img.shape[0:2] for img in imgs]
            imgs = np.array([tf.image.resize(img, model.input_shape[1:3]) for img in imgs])
            preds = model.predict(imgs)
            preds = np.array([tf.image.resize(pred, shape) for pred, shape in zip(preds, shapes)])
            for x, pred in zip(range(survey.shape[1]), preds):
                #                 img_path = f'{img_dir}{y}_{x}.tif'
                dst = f'{model_dirs[i]}pred/{roi}/{y}_{x}.tif'
                #                 write_raster(pred,img_path,dst)
                imsave(dst, pred)


def run_footprints(params):
    """ Execute estimate_footprints with parameters from params dict. """
    rois = params['rois']
    pop_rasters = params['pop_rasters']
    tile_dirs = params['tile_dirs']
    model_dirs = params['model_dirs']
    context_sizes = params['context_sizes']
    for roi, pop_path, tile_dir in zip(rois, pop_rasters, tile_dirs):
        pop = img_to_array(load_img(pop_path, color_mode="grayscale"))
        if params['survey_only']:
            estimate_footprints(roi, pop, tile_dir, model_dirs, context_sizes)
        else:
            estimate_footprints_full(roi, pop, tile_dir, model_dirs, context_sizes)
