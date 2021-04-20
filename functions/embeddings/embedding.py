import numpy as np
import os
import sys
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
sys.path.append('../swav/src/')
import resnet50 as resnet_models

from functions.embeddings.layers import LambdaLayer

tile2vec_path = './representation/tile2vec/models/boa_only.pt'

def preprocess_tilenet(tile):
    ''' Return tile preprocessed for tile2vec model '''
    pre = transforms.Compose([ # from resnet torch hub docs
        transforms.ToTensor()
    ])
    tile = Variable(pre(tile).unsqueeze(0)).cuda()
    return tile

def preprocess_imagenet(tile):
    ''' Return tile preprocess for imagenet model '''
    from torchvision import transforms
    pre = transforms.Compose([ # from resnet torch hub docs
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tile = Variable(pre(tile).unsqueeze(0)).cuda()
    return tile

def embed_survey_tiles(df,rois,tiles_path,model,model_name,preprocess_input,torch_model=False):
    ''' Embed survey tiles in df using model '''
    if torch_model:
        return embed_survey_tiles_torch(df,rois,tiles_path,model,model_name,preprocess_input)
    else:
        return embed_survey_tiles_keras(df,rois,tiles_path,model,model_name,preprocess_input)

def embed_survey_tiles_keras(df,rois,tiles_path,model,model_name,preprocess_input):
    ''' Embed survey tiles in df using keras model '''
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    input_shape = model.layers[0].input_shape[0][1:4]
    n_features = model.layers[-1].output_shape[1]
    print(f'Embedding tile to {n_features}-dimensional feature space.')
    tiles = []
    for (x,y,roi) in zip(df['x'],df['y'],df['roi']):
        f = f'{tiles_path}{roi}/images/{y}_{x}.tif'
        tile = preprocess_input(img_to_array(load_img(f,target_size=input_shape[0:2])))
        tiles.append(tile)
    tiles = np.array(tiles)
    embeddings = model.predict(tiles,verbose=1)
    for i in range(n_features):
        df[f'{model_name}_{i}'] = embeddings[:,i]
    return n_features

def embed_tile_torch(tile,model):
    ''' Embed prepared tile using torch model '''
    z = model.forward(tile)
    z = z.cpu()
    z = z.data.numpy()[0]
    return z

def embed_survey_tiles_torch(df,rois,tiles_path,model,model_name,preprocess_input,input_shape=(200,200)): # split=1)
    ''' Embed survey tiles in df using torch model '''
    model.eval() # ensure model is in evaluation mode
    embeddings = []
    n_features = -1
    for index,row in tqdm(df.iterrows(),total=len(df)):
        roi = row['roi']
        x, y = row['x'], row['y']
        f = f'./maxar_{roi}/{y}_{x}.tif'
        tile = Image.open(f).resize(input_shape)
        tile = preprocess_input(tile)
        z = embed_tile_torch(tile,model)
#         tile_master = Image.open(f).resize(input_shape)
#         split_x = input_shape[0] // split
#         split_y = input_shape[1] // split
#         embeddings_split = []
#         for i in range(0,input_shape[0],split_x):
#             for j in range(0,input_shape[1],split_y):
#                 tile = tile_master.crop((i,j,i+split_x,j+split_y))
#                 tile = preprocess_input(tile)
#                 z = embed_tile_torch(tile,model)
#                 embeddings_split.append(z)
#         z = np.sum(np.array(embeddings_split),axis=0)
        if n_features == -1:
            n_features = z.shape[0]
        embeddings.append(z)
    embeddings = np.array(embeddings)
    print(f'embedded tiles to {n_features}-dimensional feature space.')
    for i in range(n_features):
        df[f'{model_name}_{i}'] = embeddings[:,i]
    return n_features

def get_swav(pretrained=True):
    if pretrained:
        model = torch.hub.load('facebookresearch/swav', 'resnet50')
        model.fc = torch.nn.Identity() # remove fully connected portion of model
        model.cuda()
        model.eval()
        return(model,preprocess_imagenet)
    else:
        model = resnet_models.__dict__['resnet50'](
            normalize=True,
            hidden_mlp=2048,
            output_dim=128,
            nmb_prototypes=3000,
        )
        #model_fn = './representation/swav/models/boa_only_18032020/ckp-49.pth'
        model_fn = './representation/swav/models/boa_mgd_02042021/ckp-59.pth'
        old_checkpoint = torch.load(model_fn)['state_dict']
        checkpoint = OrderedDict()
        for k, v in old_checkpoint.items():
            name = k[7:]
            checkpoint[name] = v
        model.load_state_dict(checkpoint)
        model = torch.nn.Sequential(
            model,
            LambdaLayer(lambda x: x[0]) # 0 = features, 1 = cluster assignments
        )
        print(model)
        model.cuda()
        model.eval()
        return (model,preprocess_imagenet)

def get_tile2vec():
    from representation.tile2vec.src.tilenet import make_tilenet
    model = make_tilenet(in_channels=3, z_dim=512)
    model.load_state_dict(torch.load(tile2vec_path))
    model.cuda()
    model.eval()
    return(model,preprocess_tilenet)

def get_inception():
    model = torchvision.models.inception_v3(pretrained=True)
    model.fc = torch.nn.Identity() # remove fully connected portion of model
    model.cuda()
    model.eval()
    return(model,preprocess_imagenet)

def get_densenet():
    model = torchvision.models.densenet161(pretrained=True)
    model.classifier = torch.nn.Identity() # remove fully connected portion of model
    model.cuda()
    model.eval()
    return(model,preprocess_imagenet)

def get_resnet():
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity() # remove fully connected portion of model
    model.cuda()
    model.eval()
    return(model,preprocess_imagenet)

def get_vgg16():
    model = torchvision.models.vgg16_bn(pretrained=True)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1,1)) # modify average pooling to return feature vector
    model.classifier = torch.nn.Identity() # remove fully connected portion of model
    model.cuda()
    model.eval()
    return(model,preprocess_imagenet)

def get_model(model_name):
    ''' Lookup model by name '''
    if model_name == 'swav':
        return get_swav(pretrained=False)
    elif model_name == 'swav_pretrained':
        return get_swav(pretrained=True)
    elif model_name == 'tile2vec':
        return get_tile2vec()
    elif model_name == 'inception':
        return get_inception()
    elif model_name == 'densenet':
        return get_densenet()
    elif model_name == 'resnet':
        return get_resnet()
    elif model_name == 'vgg16':
        return get_vgg16()
    else:
        print("Invalid embedding model specified.")
        

def run_embeddings(df,params):
    ''' Run and save embeddings to df based on params '''
    model_names = params['models']
    rois = params['rois']
    tiles_path = params['tiles_path']
    for model_name in model_names:
        print(f'Computing {model_name} embeddings... ',end='')
        model, preprocessing = get_model(model_name)
        embed_survey_tiles_torch(df,rois,tiles_path,model,model_name,preprocessing)
    
