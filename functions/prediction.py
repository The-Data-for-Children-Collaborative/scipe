# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess
import os
from functions.scoring import meape, ameape, aggregate_percent_error
from functions.visualization import *
from tifffile import imread
from tqdm import tqdm
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import r2_score, median_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
sys.path.insert(0, os.path.abspath('.'))

# local imports
from functions.visualization import get_tiles_df, merge, prediction_error
from functions.huber import LassoHuberRegressor


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
                        return_models (:obj:`bool`, optional): Whether to return models. Defaults to False.
                        log (:obj:`bool`, optional): Whether to predict predict log of target. Defaults to False.
                
                Returns:
                        y_pred (np.ndarray): Model predictions for each row of the dataframe.
                        models (list): List of model trained on each cross validation fold returned if return_models is True.
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
        
        if huber: # TODO: improve messy workaround for R package
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
    ''' return metrics computed on observed and predicted values formatted for table '''
    return [f'{r2_score(y_true,y_pred):0.2f}',
            f'{meape(y_true,y_pred)*100:0.1f}%'.zfill(5),
            f'{ameape(y_true,y_pred):0.2f}',
            f'{median_absolute_error(y_true,y_pred):0.2f}',
            f'{aggregate_percent_error(y_true,y_pred)*100:0.1f}%'.zfill(5)]

def get_poisson():
    pr = PoissonRegressor(max_iter=200)
    reg_pr = GridSearchCV(pr,{'alpha':np.linspace(0.1,1,50)},scoring=make_scorer(median_absolute_error,greater_is_better=False),cv=3,n_jobs=-1)
    return reg_pr

def get_rf(prng):
    rf = RandomForestRegressor(n_jobs=-1,random_state=prng)
    param_grid = {'n_estimators': [100,250], # list(range(200,1200,200) 501
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]}
    reg_rf = GridSearchCV(rf,param_grid,scoring=make_scorer(median_absolute_error,greater_is_better=False),cv=3,verbose=0,n_jobs=-1)
    return reg_rf

def get_lasso():
    lr = Lasso(max_iter=200)
    reg_lr = GridSearchCV(lr,{'alpha':np.linspace(0.1,1,50)},scoring=make_scorer(median_absolute_error,greater_is_better=False),cv=3,n_jobs=-1) # np.linspace(0.1,1,50)
    return reg_lr

def get_huber():
    hr = LassoHuberRegressor()
    param_grid = {'alpha':np.linspace(0.1,1,10),
                'gamma':np.linspace(0.01,1,10)}
    reg_hr = GridSearchCV(hr,param_grid,scoring=make_scorer(median_absolute_error,greater_is_better=False),cv=3,n_jobs=-1) # np.linspace(0.1,1,50)
    return reg_hr

def get_dummy():
    dr = DummyRegressor(strategy='mean')
    reg_dr = GridSearchCV(dr,{},scoring=make_scorer(median_absolute_error,greater_is_better=False),cv=3,n_jobs=-1)
    return reg_dr

def get_model(model_name,prng):
    if model_name == 'poisson':
        return get_poisson()
    elif model_name == 'rf':
        return get_rf(prng)
    elif model_name == 'lasso':
        return get_lasso()
    elif model_name == 'huber':
        return get_huber()
    elif model_name == 'dummy':
        return get_dummy()
    else:
        print('Unidentified model, check pipeline config')
        
def run_experiments(df_master,cvs,model_names,logs,features_list,out_dir_list,experiment_dir,prng,plot_full=True,ignore_outliers_val=True,target='pop'):
    n_models = len(model_names) # number of models
    n_features = len(features_list) # number of feature sets to run each model on
    n_metrics = 5 # number of metrics to report in results table
    results = np.zeros((n_models*n_features,n_metrics*2),dtype='object') # results table
    for i,features in enumerate(features_list): # iterate through feature sets
        out_dir_master = out_dir_list[i] # where to output results of this experiment
        for j,include_outliers in enumerate([True,False]): # include/exclude outliers
            if include_outliers:
                out_dir = out_dir_master + '/outliers_included/'
            else:
                out_dir = out_dir_master + '/outliers_removed/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            df = deepcopy(df_master)
            df = df[(df['outlier']==include_outliers)|(df['outlier']==False)]
            
            models = []
            
            for cv,log,name in zip(cvs,logs,model_names): # run experiment for each model
                huber = cv == 'huber' # TODO: improve messy workaround for R package
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
            
            # update results for each model and plot feature importance
            for k,model in enumerate(model_names):
                y_pred = df[model]
                results_row = get_metrics(y_true,df[model])
                results[k*n_features+i, j*n_metrics:(j+1)*n_metrics] = results_row
                # plot error
                print('\nPlotting prediction error\n')
                prediction_error(df,true='pop',pred=model,ax=axarr_error[k],color=True) # plot
                #axarr_error[k].set_title(model)
                print('\nPlotting feature importance\n')
                if model!='huber': # TODO: improve messy workaround for R package
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
    #display(df_results)
    with open(os.path.join(experiment_dir,'table.tex'), 'w') as tf:
         tf.write(df_results.to_latex()) # write table to file
            
def get_features(csv_path):
    return np.loadtxt(csv_path, delimiter=',') 
            
def run_predictions(df,params,prng):
    model_names = params['models']
    cvs = [get_model(model,prng) for model in model_names]
    logs = params['log']
    feature_sets = [np.loadtxt(f,dtype=str) for f in params['feature_sets']]
    experiment_dir = params['experiment_dir']
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    out_dir_list = [os.path.join(experiment_dir,os.path.basename(f)[0:-4]) for f in params['feature_sets']]
    run_experiments(df,cvs,model_names,logs,feature_sets,out_dir_list,experiment_dir,prng,plot_full=False)
    
    
    
    
    