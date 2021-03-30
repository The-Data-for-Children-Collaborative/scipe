import numpy as np
import csv
from tifffile import imread
import matplotlib.pyplot as plt
plt.style.use('default')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from yellowbrick.style.palettes import LINE_COLOR
from yellowbrick.bestfit import draw_best_fit, draw_identity_line
from sklearn.metrics import r2_score, median_absolute_error
from tqdm import tqdm
import pandas as pd

from functions.scoring import meape, ameape, aggregate_percent_error

def plot_folds(df,figsize=(4.5,9),bbox=(1.75, 1)):
    ''' Return spatial plot of validation folds '''
    xs_folds = [df.loc[df['fold'] == i]['x'].values for i in range(4)]
    ys_folds = [df.loc[df['fold'] == i]['y'].values for i in range(4)]
    f, ax = plt.subplots(figsize=figsize)
    for fold in range(4):
        if fold == 3:
            ax.scatter(xs_folds[fold],ys_folds[fold],s=15,marker='s',c='purple',alpha=0.4,label=f'Fold {fold}')
        else:
            ax.scatter(xs_folds[fold],ys_folds[fold],s=15,marker='s',alpha=0.4,label=f'Fold {fold}')
    ax.invert_yaxis()
    ax.axis('scaled')
    ax.legend(loc='upper right',bbox_to_anchor=bbox)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    f.tight_layout()
    return f,ax

# returns (img,buildings) pair where img and buildings are numpy arrays representing the satellite image and building estimates corresponding to (x,y) survey cell for roi
def get_tiles(x,y,roi,tiles_path):
    ''' Return (img,buildings) pair of numpy arrays for roi survey tile (y,x) '''
    img = imread(f'{tiles_path}{roi}/images/{y}_{x}.tif')
    buildings = imread(f'{tiles_path}{roi}/buildings/{y}_{x}.tif')
    return (img,buildings)

def get_tiles_df(df,i,tiles_path):
    ''' Return (img,buildings) pair of numpy arrays for dataframe row i '''
    row = df.iloc[i]
    x,y = row['x'], row['y']
    roi = row['roi']
    return get_tiles(x,y,roi,tiles_path)

def prediction_error(df,true='pop',pred='pop_pred',ax=None,images=False,buildings=False,tiles_path=None,color=True,show_metrics=False):
    ''' Plot predicted (df[pred]) vs observed (df[true]) values from dataframe, optionally plot tile images/buildings over points '''
    # initialize plot and axis
    if not ax:
        f, ax = plt.subplots(figsize=(5,5))
    # read prediction/truth
    Y_true = df[true]
    Y_pred = df[pred]
    # plot data
    label = f"$R^2 = {r2_score(Y_true,Y_pred):0.3f}$\n$MeAPE = {meape(Y_true,Y_pred):0.3f}$\n$aMeAPE = {ameape(Y_true,Y_pred):0.3f}$\n$MeAE = {median_absolute_error(Y_true,Y_pred):0.2f}$"
    if not (images or buildings):
        if color and (not df is None):
            for i,roi in enumerate(sorted(df['roi'].unique())):
                df_roi = df[df['roi']==roi]
                ax.scatter(df_roi[true],df_roi[pred], alpha=0.75, label=roi, s=25)
        else:
            ax.scatter(Y_true, Y_pred, alpha=0.75, label=label, s=25)
            
    # draw line of best fit
    draw_best_fit(Y_true,Y_pred,ax,"linear",ls="--",lw=2,c=LINE_COLOR,label="best fit")
    
    # set the axes limits based on the range of X and Y data
    ax.set_xlim(Y_true.min() - 1, Y_true.max() + 1)
    ax.set_ylim(Y_pred.min() - 1, Y_pred.max() + 1)

    # square the axes to ensure a 45 degree line
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    # find the range that captures all data
    bounds = (min(ylim[0], xlim[0]), max(ylim[1], xlim[1]))

    # reset the limits
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)

    # ensure the aspect ratio is square
    ax.set_aspect("equal", adjustable="box")

    # draw the 45 degree line
    draw_identity_line(ax=ax,ls="--",lw=2,c=LINE_COLOR,alpha=0.5,label="identity")
    
    if show_metrics:
        ax.text(0.5, 0.98, label,
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax.transAxes)

    # set the axes labels
    ax.set_ylabel(r"Predicted population")
    ax.set_xlabel(r"Observed population")
    legend = ax.legend(loc='upper left')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0.8, 0.8, 0.8, 1e-2))
    
    ax.set_title(pred)
    
    # optionally plot images of tiles over points
    if (images or buildings) and (not df is None):
        if not tiles_path:
            print("Specify tiles path using kwarg tiles_path")
            return
        for index, row in df.iterrows():
            tiles = get_tiles_df(df,index,tiles_path)
            tile = tiles[0]
            if buildings:
                if images:
                    tile = merge(tiles[0],tiles[1])
                else:
                    tile = tiles[1]
            img = OffsetImage(tile,zoom=0.05)
            ab = AnnotationBbox(img, (row['pop'], row['pop_pred']), frameon=False)
            ax.add_artist(ab)

""" Returns latex table comparing model performance 
    dataframe df contains true values and predicted values for each model
    models is list of model names used to index the dataframe
    true is the column name of the true population value """
def get_table(df,models,true='pop'): # TODO: used?
    r2 = [f'{r2_score(df[true],df[model]):0.3g}' for model in models]
    meapes = [f'{meape(df[true],df[model])*100:0.3g}%' for model in models]
    mae = [f'{median_absolute_error(df[true],df[model]):0.2f}' for model in models]
    agg_error = [f'{aggregate_percent_error(df[true],df[model])*100:0.3g}%' for model in models]
    d = {'Model': models,'$R^2$':r2,'$MeAPE$':meapes,'$MAE$':mae,'$AgPE$':agg_error}
    df = pd.DataFrame(d)
    return df.to_latex(caption='Model performance',index=False)

def to_img(buildings,threshold=0.5):
    out = np.zeros((buildings.shape[0],buildings.shape[1],4),np.uint8)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if buildings[i,j]>threshold:
                out[i,j] = np.array((255,255,0,255))
            else: 
                out[i,j] = np.array((0,0,0,0))
    return out

def merge(image,buildings,threshold=0.5):
    image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if buildings[i,j] > threshold:
                image[i,j] = np.array([255,255,0])
    return image

def display_pair(img,buildings,axarr=None,points=None):
    if axarr is None:
        _, axarr = plt.subplots(1,3,figsize=(8,5))
    axarr[0].imshow(img)
    if points:
        xs,ys = map(lambda ls: np.array(ls),zip(*points))
        ys = (1 - ys) *img.shape[0]
        xs *= img.shape[1]
        axarr[0].scatter(xs,ys,c='red',label='Surveyed buildings')
        axarr[0].legend()
    axarr[1].imshow(img)
    axarr[1].imshow(buildings)
    axarr[2].imshow(buildings)
    
def to_pdf(survey,roi,tiles_path,out_path,points=None,coords=None,outliers=None): # returns pdf displaying population labeled survey tiles
    plt.style.use('default')
    with PdfPages(out_path) as pdf:
        with open(f'{roi}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            i = 1
            for y in tqdm(range(survey.shape[0])):
                for x in range(survey.shape[1]):
                    n = survey[y,x]
                    if n > 0:
                        if coords:
                            if (y,x) not in coords:
                                continue
                        outlier = ''
                        if outliers:
                            if (y,x) in outliers:
                                outlier = ', outlier = True'
                            else:
                                outlier = ', outlier = False'
                        img,buildings = get_tiles(x,y,roi,tiles_path)
                        f, axarr = plt.subplots(1,3,figsize=(12,5))
                        display_pair(img,to_img(buildings),axarr,points=points[y,x])
                        f.suptitle(f'Population = {n}, x = {x}, y = {y}{outlier}')
                        pdf.savefig()
                        plt.close()
                        writer.writerow([x,y,i])
                        i += 1
                        
                        
def get_colors(n,offset=0.7):
    cmap = plt.cm.get_cmap('hsv')
    colors = np.array([cmap((offset+(i)/n)%1) for i in range(n)])
    return colors

def feature_importance(models,features,colors,crop=True,n_show=10):
    if not crop:
        n_show = 35
    f, ax = plt.subplots(figsize=(n_show//2.5,5))
    importances = []
    
    model_type = type(models[0]).__name__
    y_label = ''
    if model_type == 'RandomForestRegressor':
        importances = np.array([model.feature_importances_ for model in models])
        y_label = 'Importance'
    elif model_type == 'PoissonRegressor' or model_type == 'Lasso':
        importances = np.array([model.coef_ for model in models])
        y_label = 'Coefficient Magnitude'
    elif model_type == 'DummyRegressor':
        importances = np.array([[0 for i in range(features.shape[0])] for model in models])
        y_label = 'N/A'
    else:
        print("Unsuported model")
        return
    
    # n = np.arange(importances.shape[1])
    
    means = np.mean(importances,axis=0)
    stds = np.std(importances,axis=0)
    
    if crop:
        idxs_large = np.argsort(np.absolute(means))[::-1][:n_show]
        means = means[idxs_large]
        stds = stds[idxs_large]
        features = features[idxs_large]
        colors = colors[idxs_large]
    
    n = np.arange(means.shape[0])
    
    idxs = np.argsort(means,axis=0)[::-1]
    means = means[idxs]
    stds = stds[idxs]
    labels = features[idxs]
    colors_sorted = colors[idxs]
    
    ax.bar(n,means,yerr=stds,color=colors_sorted,error_kw={'capsize':2.5,'capthick':1.0})
    ax.set_xticks(n)
    ax.set_xticklabels(labels,rotation=45,ha='right') 
    
    ax.set_ylabel(y_label)
    ax.set_xlabel('Feature')
    
    ax.set_title(f'Feature importance for {model_type}')
    
    return (f,ax)