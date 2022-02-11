""" Module for visualization of data and results related to population estimation. """

import numpy as np
import csv
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import r2_score, median_absolute_error
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scoring import meape, ameape


def plot_folds(df, figsize=(4.5, 9), bbox=(1.75, 1)):
    """ Return spatial plot of validation folds. """
    xs_folds = [df.loc[df['fold'] == i]['x'].values for i in range(4)]
    ys_folds = [df.loc[df['fold'] == i]['y'].values for i in range(4)]
    f, ax = plt.subplots(figsize=figsize)
    for fold in range(4):
        if fold == 3:
            ax.scatter(xs_folds[fold], ys_folds[fold], s=15, marker='s',
                       c='purple', alpha=0.4, label=f'Fold {fold}')
        else:
            ax.scatter(xs_folds[fold], ys_folds[fold], s=15, marker='s',
                       alpha=0.4, label=f'Fold {fold}')
    ax.invert_yaxis()
    ax.axis('scaled')
    ax.legend(loc='upper right', bbox_to_anchor=bbox)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    f.tight_layout()
    return f, ax


def get_tiles(x, y, roi, tiles_path):
    """ Return (img,buildings) pair of numpy arrays for roi survey tile
        (y,x). """
    img = imread(f'{tiles_path}{roi}/images/{y}_{x}.tif')
    buildings = imread(f'{tiles_path}{roi}/buildings/{y}_{x}.tif')
    return img, buildings


def get_tiles_df(df, i, tiles_path):
    """ Return (img,buildings) pair of numpy arrays for dataframe row i. """
    row = df.iloc[i]
    x, y = row['x'], row['y']
    roi = row['roi']
    return get_tiles(x, y, roi, tiles_path)


def prediction_error(df, true='pop', pred='pop_pred', var=None, ax=None,
                     images=False, buildings=False, tiles_path=None, color=True,
                    show_metrics=False, lim=None):
    """ Plot predicted (df[pred]) vs observed (df[true]) values from dataframe,
        optionally plot error bars (var = True) and tile images/buildings over
        points. """
    # Initialize plot and axis.
    if not ax:
        f, ax = plt.subplots(figsize=(4, 4))
    # Read prediction/truth.
    y_true = df[true]
    y_pred = df[pred]
    # Plot data.
    if not (images or buildings):
        if color:
            for i, roi in enumerate(sorted(df['roi'].unique())):
                df_roi = df[df['roi'] == roi]
                if var:
                    ax.errorbar(df_roi[true], df_roi[pred], yerr=df_roi[var],
                                capsize=0, alpha=0.3, linestyle='None')
                    ax.scatter(df_roi[true], df_roi[pred], alpha=0.5,
                               label=roi.upper(), s=25)
                else:
                    ax.scatter(df_roi[true], df_roi[pred], alpha=0.75,
                               label=roi.upper(), s=25)

    # Set the axes limits based on the range of X and Y data.
    ax.set_xlim(y_true.min() - 1, y_true.max() + 1)
    ax.set_ylim(y_pred.min() - 1, y_pred.max() + 1)

    # Square the axes to ensure a 45 degree line.
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    # Find the range that captures all data.
    bounds = (min(ylim[0], xlim[0]), max(ylim[1], xlim[1]))

    # Reset the limits.
    if lim:
        bounds = lim
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)

    # Ensure the aspect ratio is square.
    ax.set_aspect("equal", adjustable="box")

    m, b = np.polyfit(y_true, y_pred, 1)
    ax.plot(bounds, m * np.array(bounds) + b, 'k--', label='best fit',
            linewidth=2)

    # draw the 45 degree line
    plt.plot(
        bounds, bounds, '--', alpha=0.5, label='identity', color='#111111',
        linewidth=2)

    if show_metrics:
        label = (f'$R^2 = {r2_score(y_true, y_pred):0.3f}$\n$MeAPE = '
                 f'{meape(y_true, y_pred):0.3f}$\n$aMeAPE = '
                 f'{ameape(y_true, y_pred):0.3f}$\n$MeAE = '
                 f'{median_absolute_error(y_true, y_pred):0.2f}$')
        ax.text(0.5, 0.98, label,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes)

    # set the axes labels
    ax.set_ylabel(r"Predicted population")
    ax.set_xlabel(r"Observed population")
    legend = ax.legend(loc='best')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0.8, 0.8, 0.8, 0.25))

    ax.set_title(pred)

    # optionally plot images of tiles over points
    if (images or buildings) and (not df is None):
        if not tiles_path:
            print("Specify tiles path using kwarg tiles_path")
            return
        for index, row in df.iterrows():
            tiles = get_tiles_df(df, index, tiles_path)
            tile = tiles[0]
            if buildings:
                if images:
                    tile = merge(tiles[0], tiles[1])
                else:
                    tile = tiles[1]
            img = OffsetImage(tile, zoom=0.05)
            ab = AnnotationBbox(img, (row['pop'], row['pop_pred']), frameon=False)
            ax.add_artist(ab)


def to_img(buildings, threshold=0.5):
    """ Visualize building footprint estimates as image with transparent
        background where buildings <= threshold. """
    out = np.zeros((buildings.shape[0], buildings.shape[1], 4), np.uint8)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if buildings[i, j] > threshold:
                out[i, j] = np.array((255, 255, 0, 255))
            else:
                out[i, j] = np.array((0, 0, 0, 0))
    return out


def merge(image, buildings, threshold=0.5):
    """ Overlay building estimates (where > threshold) over image of tile. """
    image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if buildings[i, j] > threshold:
                image[i, j] = np.array([255, 255, 0])
    return image


def display_pair(img, buildings, axarr=None, points=None):
    """ Plot three images: img, buildings, and buildings overlayed on img. """
    if axarr is None:
        _, axarr = plt.subplots(1, 3, figsize=(8, 5))
    axarr[0].imshow(img)
    if points:
        xs, ys = map(lambda ls: np.array(ls), zip(*points))
        ys = (1 - ys) * img.shape[0]
        xs *= img.shape[1]
        axarr[0].scatter(xs, ys, c='red', label='Surveyed buildings')
        axarr[0].legend()
    axarr[1].imshow(img)
    axarr[1].imshow(buildings)
    axarr[2].imshow(buildings)


def to_pdf(survey, roi, tiles_path, out_path, points=None, coords=None,
           outliers=None):
    """ Output pdf displaying population labeled survey tiles. """
    plt.style.use('default')
    with PdfPages(out_path) as pdf:
        with open(f'{roi}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            i = 1
            for y in tqdm(range(survey.shape[0])):
                for x in range(survey.shape[1]):
                    n = survey[y, x]
                    if n > 0:
                        if coords:
                            if (y, x) not in coords:
                                continue
                        outlier = ''
                        if outliers:
                            if (y, x) in outliers:
                                outlier = ', outlier = True'
                            else:
                                outlier = ', outlier = False'
                        img, buildings = get_tiles(x, y, roi, tiles_path)
                        f, axarr = plt.subplots(1, 3, figsize=(12, 5))
                        display_pair(
                            img, to_img(buildings), axarr, points=points[y, x])
                        f.suptitle(
                            f'Population = {n}, x = {x}, y = {y}{outlier}')
                        pdf.savefig()
                        plt.close()
                        writer.writerow([x, y, i])
                        i += 1


def get_colors(n, offset=0.7):
    """ Get colormap for n features. """
    cmap = plt.cm.get_cmap('hsv')
    colors = np.array([cmap((offset + i / n) % 1) for i in range(n)])
    return colors


def feature_importance(models, features, colors, crop=True, n_show=10):
    """ Plot feature importance for model as bar chart. """
    if not crop:
        n_show = 35
    # TODO (isaac): allow custom ax to be passed.
    f, ax = plt.subplots(figsize=(n_show // 2.5, 5))

    model_type = type(models[0]).__name__
    if model_type == 'RandomForestRegressor':
        importances = np.array([model.feature_importances_ for model in models])
        y_label = 'Importance'
    elif model_type == 'PoissonRegressor' or model_type == 'Lasso':
        importances = np.array([model.coef_ for model in models])
        y_label = 'Coefficient Magnitude'
    elif model_type == 'DummyRegressor':
        importances = np.array(
            [[0 for i in range(features.shape[0])] for _ in models])
        y_label = 'N/A'
    else:
        print("Unsuported model")
        return

    means = np.mean(importances, axis=0)
    stds = np.std(importances, axis=0)

    if crop:
        idxs_large = np.argsort(np.absolute(means))[::-1][:n_show]
        means = means[idxs_large]
        stds = stds[idxs_large]
        features = features[idxs_large]
        colors = colors[idxs_large]

    n = np.arange(means.shape[0])

    idxs = np.argsort(means, axis=0)[::-1]
    means = means[idxs]
    stds = stds[idxs]
    labels = features[idxs]
    colors_sorted = colors[idxs]

    ax.bar(n, means, yerr=stds, color=colors_sorted,
           error_kw={'capsize': 2.5, 'capthick': 1.0})
    ax.set_xticks(n)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_ylabel(y_label)
    ax.set_xlabel('Feature')

    ax.set_title(f'Feature importance for {model_type}')

    return f, ax
