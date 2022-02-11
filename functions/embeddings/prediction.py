"""
Module for embedding remote sensing tiles.

Todo:
    * Speed up embedding by performing in batches.
"""
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

pd.options.mode.chained_assignment = None

from embeddings.models import get_model
from embeddings.finetuning import finetune

_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def embed_tile_torch(tile, model):
    """ Embed prepared tile using torch model. """
    z = model(tile)
    z = z.cpu()
    z = z.data.numpy()[0]
    return z


def append_file_name(df, tiles_path):
    df['file_name'] = [
        f'{tiles_path}{row.roi}/{row.y}_{row.x}.tif' for _,row in df.iterrows()]


def embed_survey_tiles(df, model, model_name, preprocessing):
    """ Embed survey tiles in df using torch model, with preprocessing function
        applied. """
    model = model.to(_device)
    model.eval()
    embeddings = []
    n_features = -1
    for _, row in tqdm(df.iterrows(), total=len(df)):
        f = row['file_name']
        tile = Image.open(f)
        tile = preprocessing(tile)
        # TODO (isaac): Could speed up by performing in batches, but df is
        # typically small.
        z = embed_tile_torch(tile, model)
        if n_features == -1:
            n_features = z.shape[0]
        embeddings.append(z)
    embeddings = np.array(embeddings)
    print(f'embedded tiles to {n_features}-dimensional feature space.')
    for i in range(n_features):
        df[f'{model_name}_{i}'] = embeddings[:, i]
    return df


def embed_survey_tiles_folds(df, models, model_name, preprocessing):
    """ Embed survey tiles in df using one torch torch model for each fold,
        with preprocessing function applied. """
    dfs = []
    for fold, model in zip(range(0, df['fold'].max()+1), models):
        dfs.append(embed_survey_tiles(df.loc[df['fold'] == fold], model,
                   model_name, preprocessing))
    df = pd.concat(dfs)
    return df


def append_precomputed(df, precomputed, rm_zero):
    """
    Append precomputed embeddings to dataset.

    Args:
        df: Dataframe to append precomputed embeddings to.
        precomputed: List of paths to precomputed embeddings (csv format).
        rm_zero (bool): Whether to remove zero population tiles from dataset
            when merging precomputed results.

    Returns:
        Dataframe with precomputed embeddings appended.

    """
    for f in precomputed:  # add precomputed embeddings to dataset
        df_pre = pd.read_csv(f, index_col=0)
        if rm_zero:
            df_pre = df_pre[df_pre['pop'] >= 1]
        df = pd.merge(df, df_pre, how='inner')
    return df


def run_embeddings(df, params, seed):
    """
    Run and save embeddings to df. Optionally append precomputed embeddings from
    disk.

    Args:
        df (pd.DataFrame): Dataframe to run embeddings over.
        params (dict): Parameters for embeddings, including models to run etc.
        seed (int): seed for finetuning (train/test split).

    Returns:
        pd.DataFrame: Dataframe with tile embeddings for each row and model.

    """
    model_names = []
    if 'models' in params:
        model_names = params['models']
    tiles_path = params['tiles_path']

    append_file_name(df, tiles_path)
    for model_name in model_names:
        model, preprocessing = get_model(model_name)
        if 'finetuning' in params and params['finetuning']['run']:
            batch_size = params['finetuning']['batch_size']
            epochs = params['finetuning']['epochs']
            freeze_epochs = params['finetuning']['frozen_epochs']
            print(f'Finetuning {model_name}... ')
            models = finetune(df, model, batch_size, epochs, freeze_epochs,
                              seed=seed)
            if params['finetuning']['save_models']:
                models_dir = params['finetuning']['models_dir']
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                print(f'Saving models to {models_dir}... ', end='')
                for fold, model in enumerate(models):
                    path = os.path.join(models_dir, f'model_{fold}.pt')
                    torch.save(model, path)
                print('done.')
            # Remove regression heads from finetuning.
            models = [model[0] for model in models]
            print(f'Computing {model_name} embeddings... ')
            df = embed_survey_tiles_folds(df, models, model_name, preprocessing)
        else:
            print(f'Computing {model_name} embeddings... ')
            df = embed_survey_tiles(df, model, model_name, preprocessing)
    return df
