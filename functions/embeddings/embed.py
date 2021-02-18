import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array

def embed_survey_tiles(df,rois,tiles_path,model,preprocess_input):
    input_shape = model.layers[0].input_shape[0][1:4]
    n_features = model.layers[-1].output_shape[1]
    print(f'Embedding tile to {n_features}-dimensional feature space')
    tiles = []
    for (x,y,roi) in zip(df['x'],df['y'],df['roi']):
        f = f'{tiles_path}{roi}/images/{y}_{x}.tif'
        tile = preprocess_input(img_to_array(load_img(f,target_size=input_shape[0:2])))
        tiles.append(tile)
    tiles = np.array(tiles)
    embeddings = model.predict(tiles,verbose=1)
    for i in range(n_features):
        df[f'embd_{i}'] = embeddings[:,i]
    return n_features
