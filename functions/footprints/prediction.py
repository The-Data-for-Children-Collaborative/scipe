import tensorflow as tf
import numpy as np
import json
import os
from tifffile import imsave, imread
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import smart_resize

from functions.data import write_raster, in_bounds

def load_model(model_path,weights_path):
    ''' Load Keras model and weights from disk. '''
    with open(model_path) as json_file:
        model_json = json.dumps(json.load(json_file))
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights_path)
        return model

def load_models(model_paths):
    ''' Load list of Keras models from disk. '''
    models = []
    for path in model_paths:
        models.append(load_model(path+'model.json',path+'model.h5'))
    return models
    
def learn_distribution(path,count):
    ''' Learn distribution of imagery by sampling subset. '''
    ls = np.array(os.listdir(path))
    n = ls.shape[0]
    idxs = np.random.randint(0, n, (count,))
    mean = np.zeros(3,)
    std = np.zeros(3,)
    print(f'Learning channel-wise distribution for imagery in {path}',flush=True)
    for i in tqdm(idxs):
        file = f'{path}{ls[i]}'
        if file.endswith('.tif'):
            img = imread(file) / 255
            mean += np.mean(img,axis=(0,1)) / count
            std += np.std(img,axis=(0,1)) / count
    return mean,std

# def resize(imgs,size=(256,256),sizes=[]):
#     if len(sizes) > 0:
#         return [tf.image.resize(img,sizes[i]) for i,img in enumerate(imgs)]
#     else:
#         return [tf.image.resize(img,size) for img in imgs]

# def standardize(imgs,mean,std):
#     return [((img/255) - mean)/std for img in imgs]

def predict_sample(img,mean,std,model):
    '''
    Standardize image with featurewise mean and standard deviation then make prediction with model.
    
            Args:
                    img (:obj:`np.ndarray`): Input image to model.
                    mean (:obj:`np.ndarray`): The feature-wise mean used to center img.
                    std (:obj:`np.ndarray`): The feature-wise standard deviation used to scale img.
                    model (:obj:`tf.keras.Model`) Keras model used for predictions.

            Returns:
                    out (:obj:`np.ndarray`): Model prediction.
    '''
    img = img/255
    original_size = img.shape[0:2]
    img = tf.image.resize(img,model.input_shape[1:3]) # resize to match model input size
    img -= mean
    img /= std
    img = np.array([img])
    out = model.predict(img)[0]
    out = tf.image.resize(out,original_size) # resize back to original size
    return out

def estimate_footprints(roi,survey,img_dir,model_paths,context_sizes,n_samples=1000):
    '''
    Estimate footprints using specified models for roi across survey tiles and surrounding context area, and save to disk.
    
            Args:
                    roi (str): The roi to predict over, used for writing to disk.
                    survey (:obj:`np.ndarray`): The survey whose tiles to predict on.
                    img_dir (str): The path to the directory containing tiles.
                    model_paths (:obj:`list` of :obj:`str`): The list of paths to models used for prediction.
                    contex_sizes (:obj:`list` of :obj:`int`): The list of context sizes to predict on around survey tiles.
                    n_samples (:obj:`int`, optional): The number of samples to use to sample featurewise mean and standard deviation.

            Returns:
                    None.
    '''
    context_size = max(context_sizes)
    mean, std = learn_distribution(img_dir,n_samples)
    print(mean,std)
    models = load_models(model_paths)
    visited = set()
    print(f'Predicting building footprints for survey tiles in {img_dir}',flush=True)
    with tqdm(total=np.sum(np.where(survey>0,1,0))) as pbar:
        for y in range(survey.shape[0]):
            for x in range(survey.shape[1]):
                if survey[y,x] > 0:
                    for x_inc in range((1-context_size)//2,(1+context_size)//2):
                        for y_inc in range((1-context_size)//2,(1+context_size)//2):
                            tile_x, tile_y = x + x_inc, y + y_inc
                            if (tile_y,tile_x) not in visited and in_bounds(survey,tile_y,tile_x):
                                img_path = f'{img_dir}{tile_y}_{tile_x}.tif'
                                img = imread(img_path)
                                for i,model in enumerate(models):
                                    dst = f'{model_paths[i]}pred/{roi}/{tile_y}_{tile_x}.tif'
                                    out = np.array(predict_sample(img,mean,std,model)).astype('float32')
                                    write_raster(out,img_path,dst)
                                visited.add((tile_y,tile_x))
                    pbar.update(1)
                    
# estimate footprints for survey tiles required by model and save to disk
# def estimate_footprints(roi,survey,imgs,model_paths,thresholds,context_sizes,n_samples=1000):
#     context_size = max(context_sizes)
#     mean, std = learn_distribution(imgs,n_samples)
#     print(mean,std)
#     models = load_models(model_paths)
#     tile_coords = []
#     tiles = []
#     print(f'Loading survey tiles in {imgs}',flush=True)
#     # load relevant tiles to run building predictions on
#     with tqdm(total=np.sum(np.where(survey>0,1,0))) as pbar:
#         for y in range(survey.shape[0]):
#             for x in range(survey.shape[1]):
#                 if survey[y,x] > 0:
#                     for x_inc in range((1-context_size)//2,(1+context_size)//2):
#                         for y_inc in range((1-context_size)//2,(1+context_size)//2):
#                             tile_x, tile_y = x + x_inc, y + y_inc
#                             if (tile_y,tile_x) not in tile_coords and in_bounds(survey,tile_y,tile_x):
#                                 img_path = f'{imgs}{tile_y}_{tile_x}.tif'
#                                 img = imread(img_path)
#                                 tile_coords.append((tile_y,tile_x))
#                                 tiles.append(img)
#                     pbar.update(1)
#     print(f'Predicting building footprints for survey tiles in {imgs}',flush=True)
#     # run and save predictions for each model
#     original_sizes = [tile.shape[0:2] for tile in tiles]
#     for i,model in enumerate(models):
#         new_tiles = standardize(resize(tiles,size=model.input_shape[1:3]),mean,std)
#         print(np.array(new_tiles).shape)
#         predictions = model.predict(new_tiles,verbose=1)
#         predictions = resize(predictions,sizes=original_sizes)                                    
#         for j in range(len(predictions)):
#             dst = f'{model_paths[i]}pred/{roi}/{tile_coords[j][0]}_{tile_coords[j][1]}.tif'
#             out = np.where(predictions[j] >= thresholds[i], 1, 0).astype('uint8') # apply threshold
#             write_raster(out,img_path,dst)