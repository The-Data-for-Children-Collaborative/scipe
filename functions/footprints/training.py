import numpy as np
import random
import os
import sys
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
np.random.seed(42)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array

from functions.visualization import to_img, display_pair
from functions.footprints.weighted_cross_entropy import weighted_cross_entropy
from functions.footprints.dice import dice_coef
from functions.footprints.vgg16_unet_model import get_vgg16

def model_from_json(path):
    '''Load model from json'''
    with open(path, 'r') as json_file:
        model = json.dumps(json.load(json_file))
        return  tf.keras.models.model_from_json(model)
    
def load_dataset(train_dir, val_dir, target_size, batch_size, ratio): # load dataset directly to memory, including <ratio> proportion of training images
    '''
    Load semantic segmentation dataset to memory in the form of training and validation iterators.
    
        Args:
                train_dir (str): The path to the directory containing training data.
                val_dir (str): The path to the directory containing validation data.
                target_size (tuple): The target size (height,width) of loaded images.
                batch_size (int): The batch size of the iterators.
                ratio (float): The percentage of data to load from the training directory.

        Returns:
                train_iterator (tf.keras.preprocessing.image.Iterator): iterator of training (image,label) pairs.
                train_size (int): number of batches in train_iterator.
                val_iterator (tf.keras.preprocessing.image.Iterator): iterator of validation (image,label) pairs.
                val_size (int): number of batches in val_iterator.
                beta (float): weight hyperparam for loss function calculated from training set.
                    
    '''
    train_dir += 'images/data/'
    val_dir += 'images/data/'
    files_train = np.array(os.listdir(train_dir))
    n = files_train.shape[0]
    idxs = np.random.randint(0, n, (int(n*ratio),)) # select random subset of images of size dictated by ratio
    
    # load training data
    X_train = np.array([img_to_array(load_img(os.path.join(train_dir,files_train[i]),target_size=target_size)) for i in tqdm(idxs,position=0,leave=True)])
    Y_train = np.array([img_to_array(load_img(os.path.join(train_dir,files_train[i]).replace('images','labels'),target_size=target_size)) for i in tqdm(idxs,position=0,leave=True)])
    beta = Y_train.size / (np.sum(Y_train) * 5)
    print(f'Loaded {idxs.shape[0]} out of {n} training images to memory', flush=True)
    
    # load validation data
    files_val = np.array(os.listdir(val_dir))
    X_val = np.array([img_to_array(load_img(os.path.join(val_dir,f),target_size=target_size)) for f in tqdm(files_val,position=0,leave=True)])
    Y_val = np.array([img_to_array(load_img(os.path.join(val_dir,f).replace('images','labels'),target_size=target_size)) for f in tqdm(files_val,position=0,leave=True)])
    print(f'Loaded {files_val.shape[0]} validation images to memory', flush=True)
    
    display_pair(X_train[10]*1./255,Y_train[10])
    plt.show()
    
    display_pair(X_val[10]*1./255,Y_val[10])
    plt.show()
    
    # initialize, fit, and apply datagen
    image_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rescale=1./255)
    image_datagen.fit(X_train)
    
    train_image_iterator = image_datagen.flow(X_train,batch_size=batch_size,shuffle=False)
    val_image_iterator = image_datagen.flow(X_val,batch_size=batch_size,shuffle=False)
    
    label_datagen = ImageDataGenerator()
    train_label_iterator = label_datagen.flow(Y_train,batch_size=batch_size,shuffle=False)
    val_label_iterator = label_datagen.flow(Y_val,batch_size=batch_size,shuffle=False)
    
    # zip and return iterators along with lengths, and beta
    train_iterator = zip(train_image_iterator,train_label_iterator)
    val_iterator = zip(val_image_iterator,val_label_iterator)
    
    # get sizes
    train_size = len(train_image_iterator)
    val_size = len(val_image_iterator)
    return train_iterator, train_size, val_iterator, val_size, beta
    
    
def sample_data(path,count,target_size):
    ''' Sample count images from path, scaled to target_size. '''
    print(f'Sampling {count} images from {path}',flush=True)
    ls = np.array(os.listdir(path))
    n = ls.shape[0]
    idxs = np.random.randint(0, n, (count,))
    xs = [img_to_array(load_img(os.path.join(path,ls[i]),target_size=target_size)) for i in tqdm(idxs)]
    return np.array(xs)

def get_image_iterator(path,X_sample,batch_size,target_size): # returns iterator for images in <path> from disk
    ''' Get image iterator loading images in path from disk, fit to X_sample. '''
    image_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rescale=1./255)   
    image_datagen.fit(X_sample)
    image_iterator = image_datagen.flow_from_directory(
        path,
        target_size=target_size,
        class_mode=None,
        batch_size=batch_size,
        shuffle=False
    ) 
    return image_iterator

def get_label_iterator(path,batch_size,target_size): # returns iterator for labels in <path> from disk
    ''' Get label iterator loading images in path from disk. '''
    label_datagen = ImageDataGenerator() 
    label_iterator = label_datagen.flow_from_directory(
        path,
        target_size=target_size,
        class_mode=None,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False
    )   
    return label_iterator

def get_iterator(path,batch_size,X_sample,target_size): # returns iterator for (image,label) pairs in <path> from disk
    ''' Get (image,label) pair iterator, fit to X_sample, loading pairs in path from disk. '''
    image_iterator = get_image_iterator(path+'images/',X_sample,batch_size,target_size)
    label_iterator = get_label_iterator(path+'labels/',batch_size,target_size)
    return (zip(image_iterator,label_iterator),len(image_iterator))

def fit_model(model,train_iterator,val_iterator,train_length,val_length,epochs,batch_size,callbacks):
    ''' Fit provided model with parameters '''
    hist = model.fit(train_iterator,
                 steps_per_epoch = train_length,
                 validation_data = val_iterator,
                 validation_steps = val_length,
                 batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks)
    return hist

""" 
    model_path specifies path of pretrained model, where applicable
    from_disk specifies if data should flow from disk (True) or be loaded into memory (False)
    ratio sets the ratio of data in train_dir to be used - only applicable if from_disk is False
                                                                                                """
def train(train_dir,val_dir,batch_size=8,epochs=1,model_path=None,target_size=(256,256),n_samples=1000,train_from_disk=False,ratio=1.0,callbacks=[]):
    '''
    Train semantic segmentation model on (image,label) pairs.
    
            Args:
                    train_dir (str): The directory containing training pairs
                    val_dir (str): The directory containing validation pairs
                    batch_size (:obj:`int`, optional) The batch size used when training. Defaults to 8.
                    epochs (:obj:`int`, optional): The number of epochs to train for. Defaults to 1.
                    model_path (:obj:`str`, optional): The path to the model to load as base for training. If no path is specified, the model is trained from scratch. Defaults to None.
                    target_size (:obj:`tuple`, optional): The target_size to load images and labels to. Defaults to (256,256).
                    n_samples (:obj:`int`, optional): The number of samples to use when determining loss weighting and featurewise mean/std of images. Defaults to 1000.
                    train_from_disk (:obj:`bool`, optional): Whether to train on data from disk (True) or from memory (False). Defaults to False.
                    ratio (:obj:`float`, optional): The percentage of data to train with. Only affects result when train_from_disk is False. Defaults to 1.0.
                    callbacks (:obj:`list`,optional): List of Keras callbacks to apply while training. Defaults to [].
                        
        Returns:
                    model (tf.keras.Model): The trained model.
                    hist (dict): The training history.
    '''
    # initialize variables to be populated during conditional
    train_iterator, train_length = None, 0
    val_iterator, val_length = None, 0
    beta = 0
    
    if train_from_disk: # load dataset from disk at runtime
        # sample data to fit datagen and compute beta
        X_sample = sample_data(train_dir+'images/data/',n_samples,target_size)
        Y_sample = sample_data(train_dir+'labels/data/',n_samples,target_size)
        beta = Y_sample.size / (np.sum(Y_sample) * 5)
        print(f'Beta: {beta:.2f}')
        # initialize iterators to flow data from disk
        train_iterator, train_length = get_iterator(train_dir,batch_size,X_sample,target_size)
        val_iterator, val_length = get_iterator(val_dir,batch_size,X_sample,target_size)
    else: # preload dataset to memory
        train_iterator, train_length, val_iterator, val_length, beta = load_dataset(train_dir,val_dir,target_size,batch_size,ratio)
    # initialize model
    model = None
    if model_path: # load model from disk
        model = model_from_json(model_path+'model.json')
        model.load_weights(model_path+'model.h5')
    else: # initialize default model
        model = get_vgg16(1, (target_size[0],target_size[1],3), batch_norm=True, dropout=True, dropout_rate=0.2, frozen=False)               
    model.compile(loss=weighted_cross_entropy(beta),optimizer='adam',metrics=[dice_coef])
    
    # train model
    hist = fit_model(model,train_iterator,val_iterator,train_length,val_length,epochs,batch_size,callbacks)
    return (model, hist)
    
    