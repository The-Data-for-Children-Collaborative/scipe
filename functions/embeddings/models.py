"""
Module containing models used to embed remote sensing tiles.
"""

import torch
import torchvision
from collections import OrderedDict

from torch.autograd import Variable
from torchvision import transforms

deepcluster_path = './representation/deepcluster/deepclusterv2_800ep_pretrain.pth.tar'
""" str: Path to deepcluster model weights. """


def get_swav():
    """ Returns (model,preprocessing) pair, where model is a PyTorch ResNet50 model pretrained on ImageNet using
    SwAV, and preprocessing is a function that preprocesses a PIL image. """
    model = torch.hub.load('facebookresearch/swav', 'resnet50')
    model.fc = torch.nn.Identity()  # remove fully connected portion of model
    model.cuda()
    model.eval()
    return model, preprocess_imagenet


def get_barlow():
    """ Returns (model,preprocessing) pair, where model is a PyTorch ResNet50 model pretrained on ImageNet using
        Barlow Twins, and preprocessing is a function that preprocesses a PIL image. """
    model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    model.fc = torch.nn.Identity()
    model.cuda()
    model.eval()
    return model, preprocess_imagenet


def get_inception():
    """ Returns (model,preprocessing) pair, where model is a PyTorch inceptionV3 model pretrained on ImageNet,
    and preprocessing is a function that preprocesses a PIL image. """
    model = torchvision.models.inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # remove fully connected portion of model
    model.cuda()
    model.eval()
    return model, preprocess_imagenet


def get_densenet():
    """ Returns (model,preprocessing) pair, where model is a PyTorch DenseNet161 model pretrained on ImageNet,
        and preprocessing is a function that preprocesses a PIL image. """
    model = torchvision.models.densenet161(pretrained=True)
    model.classifier = torch.nn.Identity()  # remove fully connected portion of model
    model.cuda()
    model.eval()
    return model, preprocess_imagenet


def get_resnet():
    """ Returns (model,preprocessing) pair, where model is a PyTorch ResNet50 model pretrained on ImageNet,
        and preprocessing is a function that preprocesses a PIL image. """
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # remove fully connected portion of model
    model.cuda()
    model.eval()
    return model, preprocess_imagenet


def get_vgg16():
    """ Returns (model,preprocessing) pair, where model is a PyTorch VGG16 model pretrained on ImageNet,
        and preprocessing is a function that preprocesses a PIL image. """
    model = torchvision.models.vgg16_bn(pretrained=True)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))  # modify average pooling to return feature vector
    model.classifier = torch.nn.Identity()  # remove fully connected portion of model
    model.cuda()
    model.eval()
    return model, preprocess_imagenet


def get_deepcluster():
    """ Returns (model,preprocessing) pair, where model is a PyTorch ResNet50 model pretrained on ImageNet using
    DeepCluster, and preprocessing is a function that preprocesses a PIL image. """
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    old_checkpoint = torch.load(deepcluster_path)
    checkpoint = OrderedDict()
    for k, v in old_checkpoint.items():
        if 'projection_head' not in k and 'prototypes' not in k:  # ignore projection head params
            name = k[7:]  # make checkpoint param names match model param names
            checkpoint[name] = v
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()
    return model, preprocess_imagenet


def get_model(model_name):
    """ Lookup model by name. """
    if model_name == 'swav':
        return get_swav()
    elif model_name == 'deepcluster':
        return get_deepcluster()
    elif model_name == 'inception':
        return get_inception()
    elif model_name == 'densenet':
        return get_densenet()
    elif model_name == 'resnet':
        return get_resnet()
    elif model_name == 'vgg16':
        return get_vgg16()
    elif model_name == 'barlow':
        return get_barlow()
    else:
        print("Invalid embedding model specified.")


def preprocess_imagenet(tile):
    """ Return tile preprocess for imagenet model. """
    pre = transforms.Compose([  # from resnet torch hub docs
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tile = Variable(pre(tile).unsqueeze(0)).cuda()
    return tile