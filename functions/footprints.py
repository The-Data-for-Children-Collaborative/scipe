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
    img = smart_resize(img,model.input_shape[1:3]) # resize to match model input size
    img -= mean
    img /= std
    img = np.array([img])
    out = model.predict(img)[0]
    out = smart_resize(out,original_size) # resize back to original size
    return out

def estimate_footprints(survey,imgs,dst,model,n_samples=1000,threshold=0.75):
    mean, std = learn_distribution(imgs,n_samples)
    print(f'Predicting building footprints for survey tiles in {imgs}',flush=True)
    with tqdm(total=np.sum(np.where(survey>0,1,0))) as pbar:
        for y in range(survey.shape[0]):
            for x in range(survey.shape[1]):
                if survey[y,x] > 0:
                    img_path = f'{imgs}{y}_{x}.tif'
                    img = imread(img_path)
                    out = predict_sample(img,mean,std,model)
                    out = np.where(out >= threshold, 1, 0).astype('uint8') # apply threshold
                    write_raster(out,img_path,f'{dst}{y}_{x}.tif')
                    pbar.update(1)