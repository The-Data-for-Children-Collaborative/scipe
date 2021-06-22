"""
Module for embedding remote sensing tiles.

Todo:
    * Speed up embedding by performing in batches.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from embeddings.models import get_model


def embed_tile_torch(tile, model):
    """ Embed prepared tile using torch model. """
    z = model.forward(tile)
    z = z.cpu()
    z = z.data.numpy()[0]
    return z


def embed_survey_tiles(df, model, model_name, preprocess_input):
    """ Embed survey tiles in df using torch model. """
    model.eval()  # ensure model is in evaluation mode
    embeddings = []
    n_features = -1
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if 'file_name' in df:
            f = row['file_name']
        else:
            roi = row['roi']
            x, y = row['x'], row['y']
            f = f'./maxar_{roi}/{y}_{x}.tif'
        tile = Image.open(f)
        tile = preprocess_input(tile)
        z = embed_tile_torch(tile, model)  # TODO: could speed up by performing in batches, but df is typically small
        if n_features == -1:
            n_features = z.shape[0]
        embeddings.append(z)
    embeddings = np.array(embeddings)
    print(f'embedded tiles to {n_features}-dimensional feature space.')
    for i in range(n_features):
        df[f'{model_name}_{i}'] = embeddings[:, i]
    return n_features


def run_embeddings(df, params, rm_zero):
    """
    Run and save embeddings to df. Optionally append precomputed embeddings from disk.

    Args:
        df (pd.DataFrame): Dataframe to run embeddings over.
        params (dict): Parameters for embeddings, including models to run etc.
        rm_zero (bool): Whether to remove zero population tiles from dataset when merging precomputed results.

    Returns:
        pd.DataFrame: Dataframe with tile embeddings for each row and model.

    """
    model_names = []
    if 'models' in params:
        model_names = params['models']
    # rois = params['rois'] TODO: remove from config file
    # tiles_path = params['tiles_path'] TODO: remove from config file
    precomputed = []
    if 'precomputed' in params:
        precomputed = params['precomputed']
    for model_name in model_names:
        print(f'Computing {model_name} embeddings... ', end='')
        model, preprocessing = get_model(model_name)
        embed_survey_tiles(df, model, model_name, preprocessing)
    for f in precomputed:  # add precomputed embeddings to dataset
        df_pre = pd.read_csv(f, index_col=0)
        if rm_zero:  # TODO: could be cleaner
            df_pre = df_pre[df_pre['pop'] >= 1]
        df = pd.merge(df, df_pre, how='inner')
    return df
