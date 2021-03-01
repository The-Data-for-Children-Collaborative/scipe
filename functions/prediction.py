# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess
import os
from functions.scoring import meape, ameape, aggregate_percent_error
from sklearn.metrics import r2_score, median_absolute_error
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
        X_train = df_train[features].to_numpy()
        Y_train = df_train[target].to_numpy().ravel()
        X_val = df_val[features].to_numpy()
        Y_val = df_val[target].to_numpy().ravel()
        
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

def get_metrics(y_true,y_pred):
    return [f'{r2_score(y_true,y_pred):0.2f}',
            f'{meape(y_true,y_pred)*100:0.1f}%'.zfill(5),
            f'{ameape(y_true,y_pred):0.2f}',
            f'{median_absolute_error(y_true,y_pred):0.2f}',
            f'{aggregate_percent_error(y_true,y_pred)*100:0.1f}%'.zfill(5)]
        
def run_experiments(df_master,cvs,model_names,logs,features_list,out_dir_list,target,prng,plot_full=True,ignore_outliers_val=True):
    n_models = len(model_names)
    n_features = len(features_list)
    n_metrics = 5
    results = np.zeros((n_models*n_features,n_metrics*2),dtype='object')
    for i,features in enumerate(features_list):
        out_dir_master = out_dir_list[i]
        for j,include_outliers in enumerate([True,False]):
            if include_outliers:
                out_dir = out_dir_master + '/outliers_included/'
            else:
                out_dir = out_dir_master + '/outliers_removed/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            df = deepcopy(df_master)
            df = df[(df['outlier']==include_outliers)|(df['outlier']==False)]
            
            models = []
            
            for cv,log,name in zip(cvs,logs,model_names): # run experiments
                huber = cv == 'huber'
                print(f'\nTraining {name}\n')
                y_pred,model =  cross_val(cv,df,features,target,log=log,return_models=True,huber=huber)
                df[name] = y_pred
                models.append(model)
            
            if ignore_outliers_val:
                df = df[df['outlier']==False] # validate without outliers
            
            y_true = df[target].to_numpy(copy=True)
    
            # Plot error, importance, and update metrics
            plt.style.use('ggplot')
            f_error, axarr_error = plt.subplots(1,n_models,figsize=(3.5*n_models,3))
            if n_models == 1:
                axarr_error = [axarr_error]
            for k,model in enumerate(model_names):
                # update metrics
                y_pred = df[model]
                results_row = get_metrics(y_true,df[model])
                results[k*n_features+i, j*n_metrics:(j+1)*n_metrics] = results_row
                # plot error
                print('\nPlotting prediction error\n')
                prediction_error(df,true='pop',pred=model,ax=axarr_error[k],color=True) # plot
                #axarr_error[k].set_title(model)
                print('\nPlotting feature importance\n')
                if model!='huber':
                    # plot importance
                    cs = get_colors(features.shape[0])
                    f,ax = feature_importance(models[k],features,cs,crop=True,n_show=15)
                    f.tight_layout(pad=1.2)
                    f.savefig(out_dir+f'{model}_importance.pdf',bbox_inches='tight')
                    if plot_full:
                        f,ax = feature_importance(models[k],features,cs,crop=False)
                        f.tight_layout(pad=1.2)
                        f.savefig(out_dir+f'{model}_importance_full.pdf',bbox_inches='tight')
            f_error.savefig(out_dir+'prediction_error.pdf',bbox_inches='tight')     
    df_results = pd.DataFrame(results)
    display(df_results)
    with open(out_dir + 'table.tex', 'w') as tf:
         tf.write(df_results.to_latex())
    
    
    