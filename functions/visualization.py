import numpy as np
import csv
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from yellowbrick.style.palettes import LINE_COLOR
from yellowbrick.bestfit import draw_best_fit, draw_identity_line
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

from functions.scoring import meape

# returns (img,buildings) pair where img and buildings are numpy arrays representing the satellite image and building estimates corresponding to (x,y) survey cell for roi
def get_tiles(x,y,roi,tiles_path):
    img = imread(f'{tiles_path}{roi}/images/{y}_{x}.tif')
    buildings = imread(f'{tiles_path}{roi}/buildings/{y}_{x}.tif')
    return (img,buildings)

def get_tiles_df(df,i,tiles_path):
    row = df.iloc[i]
    x,y = row['x'], row['y']
    roi = row['roi']
    return get_tiles(x,y,roi,tiles_path)

def prediction_error(Y_true,Y_pred,ax=None,images=False,buildings=False,df=None,tiles_path=None,color=False): # plot predicted (Y_pred) vs observed (Y_true)
    # initialize plot and axis
    if not ax:
        f, ax = plt.subplots(figsize=(5,5))
    # plot data
    label = f"$R^2 = {r2_score(Y_true,Y_pred):0.3f}$\n$MeAPE = {meape(Y_true,Y_pred):0.3f}$\n$MAE = {mean_absolute_error(Y_true,Y_pred):0.2f}$"
    if not (images or buildings):
        if color and (not df is None):
            for i,roi in enumerate(df['roi'].unique()):
                df_roi = df[df['roi']==roi]
                if i == 0:
                    ax.scatter(df_roi['pop'],df_roi['pop_pred'], alpha=0.75, label=label, s=25)
                else:
                    ax.scatter(df_roi['pop'],df_roi['pop_pred'], alpha=0.75, s=25)
        else:
            ax.scatter(Y_true, Y_pred, alpha=0.75, label=label, s=25)
            
    # draw line of best fit
    draw_best_fit(Y_true,Y_pred,ax,"linear",ls="--",lw=2,c=LINE_COLOR,label="best fit")
    
    # Set the axes limits based on the range of X and Y data
    ax.set_xlim(Y_true.min() - 1, Y_true.max() + 1)
    ax.set_ylim(Y_pred.min() - 1, Y_pred.max() + 1)

    # Square the axes to ensure a 45 degree line
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    # Find the range that captures all data
    bounds = (min(ylim[0], xlim[0]), max(ylim[1], xlim[1]))

    # Reset the limits
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)

    # Ensure the aspect ratio is square
    ax.set_aspect("equal", adjustable="box")

    # Draw the 45 degree line
    draw_identity_line(ax=ax,ls="--",lw=2,c=LINE_COLOR,alpha=0.5,label="identity")

    # Set the axes labels
    ax.set_ylabel(r"$\hat{y}$")
    ax.set_xlabel(r"$y$")
    ax.legend()
    
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

def display_pair(img,buildings,axarr):
    axarr[0].imshow(img)
    axarr[1].imshow(img)
    axarr[1].imshow(buildings)
    axarr[2].imshow(buildings)
    
def to_pdf(survey,count,roi,tiles_path,out_path):
    with PdfPages(out_path) as pdf:
        with open(f'{roi}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            i = 1
            for y in tqdm(range(survey.shape[0])):
                for x in range(survey.shape[1]):
                    n = survey[y,x]
                    if n > 0:
                        img,buildings = get_tiles(x,y,roi,tiles_path)
                        f, axarr = plt.subplots(1,3,figsize=(8,3))
                        display_pair(img,to_img(buildings),axarr)
                        f.suptitle(f'Population = {n}, x = {x}, y = {y}, count = {count[y,x]}')
                        pdf.savefig()
                        plt.close()
                        writer.writerow([x,y,i])
                        i += 1