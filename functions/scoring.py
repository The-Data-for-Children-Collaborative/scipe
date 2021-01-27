import numpy as np
from sklearn.metrics import accuracy_score

def to_categorical(ys,bands):
    def categorize(y):
        for i,band in enumerate(bands):
            if band[0] < y <= band[1]:
                return i
    ys_cat = []
    for y in ys:
        ys_cat.append(categorize(y))
    return np.array(ys_cat)

def accuracy(y_true,y_pred,bands):
    y_true = to_categorical(y_true,bands)
    y_pred = to_categorical(y_pred,bands)
    return accuracy_score(y_true,y_pred)

def meape(y_true,y_pred):
    abs_percentage_errors = []
    for true,pred in zip(y_true,y_pred):
        abs_percentage_errors.append(abs((true-pred)/true))
    return np.median(abs_percentage_errors)

def ameape(y_true,y_pred,k=10):
    abs_percentage_errors = []
    for true,pred in zip(y_true,y_pred):
        abs_percentage_errors.append(abs((true-pred)/(true+k)))
    return np.median(abs_percentage_errors)

def aggregate_percent_error(y_true,y_pred):
    total_true = np.sum(y_true)
    total_pred = np.sum(y_pred)
    return abs(total_pred-total_true)/total_true