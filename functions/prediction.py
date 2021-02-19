# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess
import os
from tifffile import imread
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
sys.path.insert(0, os.path.abspath('.'))

# local imports
from functions.visualization import get_tiles_df, merge, prediction_error

def pop_histogram(Y,ax,y_label):
    ''' Plot histogram of population '''
    ax.hist(Y,50,(0,100))
    ax.set_xlabel('Population')
    ax.set_ylabel(y_label)
    ax.set_ylim(0,20)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable="box")
    
def get_split(df,k):
    ''' Return training/validation split for fold k in df '''
    df_train = df[df['fold'] != k].drop('fold',axis=1)
    df_val = df[df['fold'] == k].drop('fold',axis=1)
    ratio = len(df_val)/(len(df_val)+len(df_train))
    print(f'Training on {len(df_train)} samples, validating on {len(df_val)}, {(1-ratio)*100:.0f}/{ratio*100:.0f} split')
    return df_train, df_val
    
def cross_val(reg_master,df,features,target,return_models=True,log=False,huber=False):
    ''' Train regression model using cross-validation on dataframe pre-split into folds 
                Args:
                        reg_master (sklearn.model_selection.GridSearchCV): A grid search instance to be trained.
                        df (pd.DataFrame): The dataframe used for training and validation.
                        features (:obj:`list` of :obj:`str`): The dataframe columns used as features during training.
                        target (str): The dataframe column used as target variable during training.
                        return_models (:obj:`bool`, optional): Whether or not to return models. Defaults to False.
                        log (:obj:`bool`, optional): If True predict log of target. Defaults to False.
                
                Returns:
                        y_pred (np.ndarray): Model predictions for each row of the dataframe.
                        models (list): List of model trained on each cross validation fold. Only returned if return_models is True.
    '''
    ks = np.sort(df['fold'].unique()) # list of folds specified in dataframe
    if len(ks) == 0:
        print("No folds specified in dataframe")
        return
    y_pred = []
    models = []
    for i in ks:
        # start with fresh grid search instance 
        reg = deepcopy(reg_master)
        
        # get train/val split for this fold
        df_train, df_val = get_split(df,i)
        
        # convert dataframes to numpy arrays
        X_train = df_train[features].to_numpy(copy=True)
        Y_train = df_train[target].to_numpy(copy=True).ravel()
        X_val = df_val[features].to_numpy(copy=True)
        Y_val = df_val[target].to_numpy(copy=True).ravel()
        
        if log:
            Y_train = np.log(Y_train)
            Y_val = np.log(Y_val)
        
        # initialize scaler, fit, and scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        model = None
        
        if huber: # messy workaround
            y_pred += list(huber_regression(X_train,Y_train,X_val,Y_val))
        else:
            # fit model with grid search
            gs = reg.fit(X_train, Y_train)
            model = gs.best_estimator_
            print(reg.best_params_)
            y_pred += list(model.predict(X_val))
            #print(model.intercept_)
 
        # append predictions and model
        models.append(model)
    if log:
        y_pred = np.exp(y_pred)
    if return_models:
        return (np.array(y_pred), models)
    else:
        return np.array(y_pred)
    
def huber_regression(X_train,y_train,X_val,y_val):
    # save data to file
    np.savetxt('./csv/X_train.csv', X_train, delimiter=',') 
    np.savetxt('./csv/y_train.csv', y_train, delimiter=',')
    np.savetxt('./csv/X_val.csv', X_val, delimiter=',')
    np.savetxt('./csv/y_val.csv', y_val, delimiter=',')
    
    # run huber regression in R
    subprocess.run(['rscript', './functions/run_huber.r'],check=False)
    
    # load predictions
    y_pred = np.loadtxt('./csv/y_pred.csv', delimiter=',', skiprows=1)
    
    # clean up
    os.remove('./csv/X_train.csv')
    os.remove('./csv/y_train.csv')
    os.remove('./csv/X_val.csv')
    os.remove('./csv/y_val.csv')
    os.remove('./csv/y_pred.csv')
    
    return y_pred
    
    