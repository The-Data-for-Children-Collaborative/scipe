import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold
from embeddings.prediction import embed_tile_torch
from PIL import Image, ImageOps

SEED = 42
""" int: Seed to control randomness. """
np.random.seed(SEED)
prng = np.random.RandomState(SEED)
""" np.random.RandomState: Numpy random state based on SEED. """


def plot_tiles(embds, tiles, pops=None, zoom=0.1):
    """ Plot 2D tile embeddings, optionally coloured by list of population
        labels.
    """
    if pops is None:
        pops = []
    cmap = plt.cm.get_cmap('plasma')  # color map for population
    f, ax = plt.subplots(1, figsize=(10, 10))
    ax.scatter(embds[:, 0], embds[:, 1])
    for i in range(embds.shape[0]):
        x, y = embds[i]
        tile = tiles[i]
        if pops:
            c = cmap(pops[i])
            c = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            tile = ImageOps.expand(tile, border=20, fill=c)
        img = OffsetImage(tile, zoom=zoom)
        ab = AnnotationBbox(img, (x, y), frameon=False)
        ax.add_artist(ab)
    ax.axis('off')
    return f, ax


def interpolate_feature(model, tiles_master, preprocessing, dim, n=5):
    """ Return n indices that interpolate through tiles in specified dimension
        of embedding. """
    tiles = np.array([preprocessing(tile) for tile in tiles_master])
    y_pred = np.array([embed_tile_torch(tile, model)
                       for tile in tqdm(tiles, position=0, leave=True)])
    y_pred = y_pred[:, dim]
    # Descending order to ensure largest value of dim is returned.
    idxs = np.argsort(y_pred)[::-1]
    # Reverse to give ascending order.
    return [idxs[i] for i in range(0, idxs.shape[0] - 1,
            idxs.shape[0] // n)][::-1]


def reduce_tsne(model, tiles, preprocessing):
    """Reduce model embeddings over tiles with TSNE."""
    # Filter black tiles.
    tiles = np.array(
        [preprocessing(tile) for tile in tiles if np.sum(np.array(tile)) > 0])
    y_pred = [embed_tile_torch(tile, model) for
              tile in tqdm(tiles, position=0, leave=True)]
    y_pred = np.array(y_pred)
    return manifold.TSNE(n_components=2, verbose=1, n_jobs=-1, init='pca',
                         random_state=prng).fit_transform(y_pred)


def visualize_embeddings(model, tiles_master, preprocessing, zoom=0.1):
    """ Visualize model embeddings of tiles using T-SNE. """
    reduced = reduce_tsne(model, tiles_master, preprocessing)
    return plot_tiles(reduced, tiles_master, zoom=zoom)


def visualize_embeddings_df(model, df, preprocessing, input_shape, zoom=0.1):
    """ Visualize model embeddings of tiles in dataframe df using T-SNE. """
    min_pop, max_pop = df['pop'].min(), df['pop'].max()
    tiles_master = []
    tiles = []
    pops = []
    for _, row in df.iterrows():
        tile = Image.open(
            f'./survey_tiles/{row.roi}/images/{row.y}_{row.x}.tif').resize(
                input_shape)
        tiles_master.append(tile)
        tiles.append(preprocessing(tile))
        # Scale to [0,1].
        pop = (row['pop'] - min_pop) / (max_pop - min_pop)
        pops.append(pop)
    y_pred = np.array([embed_tile_torch(tile, model)
                       for tile in tqdm(tiles, position=0, leave=True)])
    reduced = manifold.TSNE(n_components=2, verbose=1, n_jobs=-1, init='pca',
                            random_state=prng).fit_transform(y_pred)
    return plot_tiles(reduced, tiles_master, pops=pops, zoom=zoom)
