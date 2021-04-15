import yaml
import sys
import os
import numpy as np
from shutil import copyfile
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath('.'))
from functions.preprocessing import preprocess_data
from functions.data import build_dataset
from functions.prediction import run_predictions, run_estimation
from functions.embeddings.embedding import run_embeddings

SEED = 42
np.random.seed(SEED)
prng = np.random.RandomState(SEED)
plt.style.use('ggplot')

params = {}
with open(sys.argv[1]) as file:
    params = yaml.safe_load(file)

if params['preprocessing']['run']:
    preprocess_data(params['preprocessing'])
else:
    print("Skipping preprocessing")
    
df = build_dataset(params['dataset'])

if params['embedding']['run'] or (params['estimation']['run'] and params['estimation']['embed']):
    run_embeddings(df,params['embedding'])
else:
    print("Skipping embeddings")

if params['prediction']['run']:
    run_predictions(df,params['prediction'],prng)
else:
    print("Skipping survey predictions")
    
if params['estimation']['run']:
    params_dataset = params['dataset'].copy()
    params_dataset['rois'] = params['estimation']['rois'] # may not want to predict over all rois from modelling stage
    df_full = build_dataset(params_dataset,survey_only=False)
    if params['estimation']['embed']: # rerun same embeddings from experiments on full dataset
        run_embeddings(df_full,params['embedding'])
    run_estimation(df,df_full,params['estimation'],prng)
    

exp_dir = params['prediction']['experiment_dir']
src = sys.argv[1]
dst = os.path.join(exp_dir,os.path.basename(src))
copyfile(src,dst) # copy params to experiment dir 