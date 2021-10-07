""" Script for running population estimation pipeline.

Usage:
    ``python run_pipeline.py <path_to_config>`` (run from parent directory)
"""

import yaml
import sys
import os
import numpy as np
from shutil import copyfile
from matplotlib import pyplot as plt

from joblib import parallel_backend
sys.path.append(os.path.abspath('./functions/'))  # assumes running from parent directory
from population.preprocessing import preprocess_data
from population.data import build_dataset
from population.prediction import run_predictions, run_estimation
from embeddings.prediction import run_embeddings, append_precomputed

SEED = 42
np.random.seed(SEED)
prng = np.random.RandomState(SEED)
plt.style.use('ggplot')

if __name__ == "__main__":
    with parallel_backend('threading', n_jobs=-1):
        with open(sys.argv[1]) as file:
            params = yaml.safe_load(file)

        if params['preprocessing']['run']:
            preprocess_data(params['preprocessing'])
        else:
            print("Skipping preprocessing")

        df = build_dataset(params['dataset'])

        if params['embedding']['run'] or (params['estimation']['run'] and params['estimation']['embed']):
            df = run_embeddings(df, params['embedding'], SEED)
        else:
            print("Skipping running embeddings")
        if params['embedding']['append_precomputed']:
            print('Appending precomputed embeddings... ', end='')
            precomputed = params['embedding']['precomputed']
            df = append_precomputed(df, precomputed, 'zero_label_paths' not in params['dataset'])
            print('done.')

        if params['dataset']['save_dataset']:
            dir, _ = params['dataset']['dataset_path'].rsplit('.', 1)
            if not os.path.exists(dir):
                os.makedirs(dir)
            df.to_csv(params['dataset']['dataset_path'])

        if params['prediction']['run']:
            run_predictions(df, params['prediction'], prng)
        else:
            print("Skipping survey predictions")

        if params['estimation']['run']:
            params_dataset = params['dataset'].copy()
            params_dataset['rois'] = params['estimation'][
                'rois']  # may not want to predict over all rois from modelling stage
            df_full = build_dataset(params_dataset, survey_only=False)
            if params['estimation']['embed']:  # rerun same embeddings from experiments on full dataset
                df_full = run_embeddings(df_full, params['embedding'])
            run_estimation(df, df_full, params['estimation'], prng)

        exp_dir = params['prediction']['experiment_dir']
        src = sys.argv[1]
        dst = os.path.join(exp_dir, os.path.basename(src))
        copyfile(src, dst)  # copy params to experiment dir
