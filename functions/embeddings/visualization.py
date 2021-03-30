import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tifffile import imread
from sklearn.manifold import TSNE
from functions.embeddings.embedding import embed_tile_torch
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from PIL import Image, ImageOps

SEED = 42
np.random.seed(SEED)
prng = np.random.RandomState(SEED)

def plot_tiles(embds,tiles,pops=[],pop_range=(0,1),zoom=0.1,figsize=(20,20)):
    ''' Plot 2D tile embeddings, optionally coloured by list of population labels '''
    cmap = plt.cm.get_cmap('plasma') # color map for population
    f,ax = plt.subplots(1,figsize=(10,10))
    ax.scatter(embds[:,0],embds[:,1])
    for i in range(embds.shape[0]):
        x,y = embds[i]
        tile = tiles[i]
        if pops:
            c = cmap(pops[i])
            c = (int(c[0]*255),int(c[1]*255),int(c[2]*255))
            #print(c)
            tile = ImageOps.expand(tile,border=20,fill=c)
        img = OffsetImage(tile,zoom=zoom)
        ab = AnnotationBbox(img, (x, y), frameon=False)
        ax.add_artist(ab)
    return f,ax

def visualize_embeddings(model,tiles_master,preprocessing,torch_model=True,zoom=0.1):
    ''' Visualize model embeddings of tiles using T-SNE '''
    y_pred = []
    tiles = np.array([preprocessing(tile) for tile in tiles_master])
    if torch_model:
        y_pred = [embed_tile_torch(tile,model) for tile in tqdm(tiles,position=0,leave=True)]
    else: # assume keras model
        y_pred = model.predict(tiles,verbose=1)
    y_pred = np.array(y_pred)
    reduced = TSNE(n_components=2,verbose=1,n_jobs=-1,init='pca',random_state=prng).fit_transform(y_pred)
    return plot_tiles(reduced,tiles_master,zoom=zoom)

def visualize_embeddings_df(model,df,preprocessing,input_shape,torch_model=True,zoom=0.1):
    ''' Visualize model embeddings of tiles in dataframe df using T-SNE '''
    min_pop, max_pop = df['pop'].min(), df['pop'].max()
    tiles_master = [] 
    tiles = []
    pops = []
    y_pred = []
    for index, row in df.iterrows():
        tile = Image.open(f'./survey_tiles/{row.roi}/images/{row.y}_{row.x}.tif').resize(input_shape)
        tiles_master.append(tile)
        tiles.append(preprocessing(tile))
        pop = (row['pop'] - min_pop) / (max_pop - min_pop)
        pops.append(pop)
    if torch_model:
        y_pred = [embed_tile_torch(tile,model) for tile in tqdm(tiles,position=0,leave=True)]
    else: # assume keras model
        y_pred = model.predict(tiles,verbose=1)
    y_pred = np.array(y_pred)
    reduced = TSNE(n_components=2,verbose=1,n_jobs=-1,init='pca',random_state=prng).fit_transform(y_pred)
    return plot_tiles(reduced,tiles_master,zoom=zoom,pops=pops,pop_range=(min_pop,max_pop))