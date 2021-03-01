import numpy as np
import os
from sklearn.base import BaseEstimator, RegressorMixin
from functions.prediction import fit_huber, predict_huber
import shutil

class LassoHuberRegressor(BaseEstimator,RegressorMixin):

    def __init__(self, alpha=0.1, gamma=0.1):
        self.alpha = alpha # regularization strength
        self.gamma = gamma # sensitivity to outliers
        self.disk_loc = f'./huber/{np.random.randint(0,1e9)}/' # location on disk
        
        #print(f"New regressor in {self.disk_loc} with alpha = {self.alpha}, gamma = {self.gamma}")
        
        if not os.path.exists(self.disk_loc):
            os.makedirs(self.disk_loc)
            
    def __del__(self):
        # return
        try:
            if os.path.exists(self.disk_loc):
                shutil.rmtree(self.disk_loc) # doesn't always work, fine for now
        except:
            pass

    def fit(self, X, y):
        # save data to file
        np.savetxt(os.path.join(path,'X_train.csv'), X, delimiter=',') 
        np.savetxt(os.path.join(path,'y_train.csv'), y, delimiter=',')
    
        # fit huber regressor in R
        subprocess.call(f'rscript ./functions/fit_huber.r  {str(self.gamma)} {self.disk_loc}')
        
    def predict(self, X):
        # save data to file
        np.savetxt(os.path.join(path,'X_pred.csv'), X, delimiter=',') 

        # predict with huber regression in R
        subprocess.call(f'rscript ./functions/predict_huber.r  {str(self.alpha)} {self.disk_loc}')

        # load predictions
        y_pred = np.loadtxt(os.path.join(path,'y_pred.csv'), delimiter=',', skiprows=1)
        
        return y_pred
    
