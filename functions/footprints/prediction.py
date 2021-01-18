import tensorflow as tf
import numpy as np
import json
import os
from tifffile import imsave, imread
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import smart_resize

from functions.data import write_raster

def load_model(model_path,weights_path):
    with open(model_path) as json_file:
        model_json = json.dumps(json.load(json_file))
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights_path)
        return model

def load_models(model_paths):
    models = []
    for path in model_paths:
        models.append(load_model(path+'model.json',path+'model.h5'))
    return models
    
def learn_distribution(path,count):
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

def predict_sample(img,mean,std,model):
    img = img/255
    original_size = img.shape[0:2]
    img = tf.image.resize(img,model.input_shape[1:3]) # resize to match model input size
    img -= mean
    img /= std
    img = np.array([img])
    out = model.predict(img)[0]
    out = tf.image.resize(out,original_size) # resize back to original size
    return out

def estimate_footprints(roi,survey,imgs,model_paths,thresholds,n_samples=1000):
    mean, std = learn_distribution(imgs,n_samples)
    print(mean,std)
    models = load_models(model_paths)
    print(f'Predicting building footprints for survey tiles in {imgs}',flush=True)
    with tqdm(total=np.sum(np.where(survey>0,1,0))) as pbar:
        for y in range(survey.shape[0]):
            for x in range(survey.shape[1]):
                if survey[y,x] > 0:
                    img_path = f'{imgs}{y}_{x}.tif'
                    img = imread(img_path)
                    for i,model in enumerate(models):
                        dst = f'{model_paths[i]}pred/{roi}/{y}_{x}.tif'
                        out = predict_sample(img,mean,std,model)
                        out = np.where(out >= thresholds[i], 1, 0).astype('uint8') # apply threshold
                        write_raster(out,img_path,dst)
                    pbar.update(1)