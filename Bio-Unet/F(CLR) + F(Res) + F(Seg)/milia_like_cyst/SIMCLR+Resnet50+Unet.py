#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.executable


# In[2]:


import os
import re
import time
import copy
import h5py
import random
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
from PIL import Image


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[4]:


import torchvision.models
from torchvision import transforms
from torchvision.models.resnet import resnet50


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score


# In[6]:


import captum.attr as cattr
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLinearRegression,  SkLearnLasso


# In[7]:


from monai.losses.dice import DiceLoss, DiceCELoss


# # args

# In[8]:


ATTR_TO_INDEX = {
    'globules' : 0,
    'milia_like_cyst' : 1,
    'negative_network' : 2,
    'pigment_network' : 3,
    'streaks' : 4,
}

INDEX_TO_ATTR = { idx:attr for idx, attr in ATTR_TO_INDEX.items() }


# In[9]:


parser = argparse.ArgumentParser()

parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')

parser.add_argument('--report-freq', type=int, default=1, help='logging frequency')
parser.add_argument('--tune-mode', type=str, default='fine-tune', choices=['fine-tune', 'feature-extract'], help='tuning mode' )
parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'vgg19', 'inception_v3'], 
                    help='backbone architecture' )
parser.add_argument('--cls-type', type=str, default='single', choices=['single', 
                   'double', 'double-bn', 'double-dropout'], help='classifier architecture' )
parser.add_argument('--hidden-dim', type=int, default=512, help='hidden dimension of classifier' )
parser.add_argument('--record-root-dir', type=str, default='./record-data', help='record data root dir' )
parser.add_argument('--exp', type=str, default='default_exp', help='name of experiment' )
parser.add_argument('--batch-size', type=int, default=8, help='batch size' )
parser.add_argument('--UNet-batch-size', type=int, default=8, help='UNet batch size' )
parser.add_argument('--num-workers', type=int, default=4, help='number of processes working on cpu.')
parser.add_argument('--num-classes', type=int, default=5, help='number of classes')
parser.add_argument('--num-epochs', type=int, default=20,  help='number of epochs.')
parser.add_argument( '--num-steps', type=int, default=-1, help='number of steps per epoch. '+ '-1 means use entire data' )
parser.add_argument('--learn-rate', type=float, default=1e-3, help='learning rate for gradient descent')
parser.add_argument('--weight-decay', type=float, default=1e-3, help='weight decay for optimization')
parser.add_argument('--resume', action='store_true', help='resume experiment <exp> from last checkpoint' )
parser.add_argument('--input-dir1', type=str, default= r'', help='data root dir' )
parser.add_argument('--input-dir2', type=str, default= r'', help='data root dir' )
parser.add_argument('--input-dir3', type=str, default= r'', help='data root dir' )
parser.add_argument('--save-name', type=str, default='model.pt', help='saved model name' )
parser.add_argument('--DEVICE', type=str, default='cpu', help='cuda' )

args, unknown =  parser.parse_known_args()


# In[10]:


# device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
args.device = DEVICE
args.gpu_ids = [0]

ntimes = 2
args.num_epochs = 100
args.learn_rate = 1e-6
args.weight_decay = 1e-8 
args.batch_size = 16 * ntimes
args.UNet_batch_size = 8 * ntimes
args.unsup_batch_size = 24 * ntimes

feature_dim, temperature, k = args.feature_dim, args.temperature, args.k


# In[11]:


args.root = '/scratch/rs37890/CARC/Explainable-NN-model'
args.sub_root = '/Fold1_Bio-Unet-stage1/F(CLR) + F(Res) + F(Seg)'
args.attribute = '/milia_like_cyst'
args.attribute_name = 'milia_like_cyst'
args.attribute_fullname = 'milia_like_cyst'
args.attribute_shortname = 'milia_like_cyst'

args.csv_h5_dir = args.root + '/Data'
args.dataset = args.root + '/Dataset'

args.generate_Heatmap_dir = args.root  + args.sub_root + args.attribute + '/Generate_Heatmap_bestmodel'
args.H5py_PT = args.root  + args.sub_root + args.attribute + '/H5py&PT'

args.record_dir = args.root  + args.sub_root + args.attribute + '/record_dir'
args.save_name = 'SIMCLR+Resnet50+Unet'


# # dataset

# In[12]:


def get_transform(split):
    if split == 'unsup':
        img_size = ( 512, 512 )
        unsup_transform = transforms.Compose([ transforms.Resize( img_size ), 
                                               transforms.RandomResizedCrop(512), #img size
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                               transforms.RandomGrayscale(p=0.2),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        return unsup_transform
    
    else:
        img_size = ( 512, 512 )
        Normal_transform = transforms.Compose([ transforms.Resize( img_size ),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])],
                                             )
    
        return Normal_transform


# In[13]:


class SkinDataset( torch.utils.data.Dataset ):

    def __init__( self, h5_file_path, transform, split, args):
        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.transform = transform
        self.split = split
        self.img_id_to_h5idx = self.build_img_id_to_h5idx()
        self.num_imgs = self.get_num_imgs() 

        self.args = args

        self.ATTR_TO_INDEX = { 'globules': 0,
                               'milia_like_cyst': 1,
                               'negative_network': 2,
                               'pigment_network': 3,
                               'streaks': 4,
                             }

    def get_num_imgs( self ):
        with h5py.File( self.h5_file_path, 'r' ) as f:
            return len( f['image_ids'] )

    def build_img_id_to_h5idx( self ):
        with h5py.File( self.h5_file_path, 'r' ) as f:
            
            img_ids = f['image_ids']
            img_id_to_h5idx = { img_id : idx for idx, img_id in enumerate( img_ids ) }
            
            return img_id_to_h5idx

    def __len__( self ):
        return self.get_num_imgs()

    def __getitem__( self, idx ):
        # import pdb; pdb.set_trace()
        if not self.h5_file:
            self.h5_file = h5py.File( self.h5_file_path, 'r' )
            
        img_id = self.h5_file['image_ids'][idx]
        img = self.h5_file['images'][idx]
        img = img.transpose([1, 2, 0])

        assert img.shape == (512, 512, 3)

        img = Image.fromarray( np.uint8(img) )
        
        if self.split == 'unsup':
            img1 = self.transform( img )
            img2 = self.transform( img )
            return img_id, img1, img2
        
        else:
            masks = self.h5_file['masks'][idx]
            labels = self.h5_file['labels'][idx].astype(np.float64)
            img = self.transform( img )

            # labels[0] globus
            # labels[1] milia_like_cyst
            # labels[2] negative
            # labels[3] pigment
            # labels[4] streaks

            index = self.ATTR_TO_INDEX[args.attribute_fullname]
            index = int(index)

            assert masks.shape == (5, 512, 512)
            assert np.expand_dims(labels[index], axis=0).shape == (1,)
            
            return img_id, img, np.expand_dims(labels[index], axis=0), masks


# In[14]:


def get_dataloader( args ):
    
    dataloader = {}
    splits = [ 'unsup', 'train', 'val' ]
    
    for split in splits:
        
        h5_file_path = os.path.join( args.csv_h5_dir, f'{split}.h5' )
        transform = get_transform(split)

        if split == 'unsup':
            bsz = args.unsup_batch_size
        else:
            bsz = args.batch_size
        
        
        dataset = SkinDataset( h5_file_path, transform, split, args)
        loader = torch.utils.data.DataLoader( dataset = dataset,
                                              batch_size = bsz,
                                              shuffle = True if split != 'val' else False,
                                              num_workers=args.num_workers,
                                              drop_last = True if split == 'unsup' else False,
                                            )
        
        dataloader[ split ] = loader
        
    return dataloader


# # Skin_Heatmaps_Dataset

# In[15]:


def get_heatmaps_transform():
    img_size = ( 512, 512 )
    transform = transforms.Compose( [
        transforms.ToTensor(),
        ] )
    return transform


# In[16]:


class Skin_Heatmaps_Dataset( torch.utils.data.Dataset ):

    def __init__( self, h5_file_path, transform ):
        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.transform = transform
        self.img_id_to_h5idx = self.build_img_id_to_h5idx()
        self.num_imgs = self.get_num_imgs() 

    def get_num_imgs( self ):
        with h5py.File( self.h5_file_path, 'r' ) as f:
            return len( f['image_ids'] )

    def build_img_id_to_h5idx( self ):
        with h5py.File( self.h5_file_path, 'r' ) as f:
            img_ids = f['image_ids']
            img_id_to_h5idx = \
                    { img_id : idx for idx, img_id in enumerate( img_ids ) }
            return img_id_to_h5idx

    def __len__( self ):
        return self.get_num_imgs()

    def __getitem__( self, idx ):
        # import pdb; pdb.set_trace()
        if not self.h5_file:
            self.h5_file = h5py.File( self.h5_file_path, 'r' )
        img_id = self.h5_file['image_ids'][idx]
        img = self.h5_file['images'][idx]
        masks = self.h5_file['masks'][idx]
        heatmaps = self.h5_file['heatmaps'][idx]
        labels = self.h5_file['labels'][idx].astype(np.float64)
        
        img = img.transpose([1, 2, 0])
        img = Image.fromarray( np.uint8(img) )
        
        heatmaps = heatmaps.transpose([1, 2, 0])

        if self.transform:
            img = self.transform( img )
            heatmaps = self.transform( np.uint8(heatmaps) )

        return img_id, img, labels, masks, heatmaps


# In[17]:


def get_heatmap_dataloader( args ):
    
    dataloader = {}
    splits = [ 'train', 'val' ]
    for split in splits:
        h5_file_path = os.path.join( args.generate_Heatmap_dir, f'{split}_heatmap.h5' )
        transform = get_heatmaps_transform()
        dataset = Skin_Heatmaps_Dataset( h5_file_path, transform )
        
        loader = torch.utils.data.DataLoader( dataset=dataset,
                                              batch_size=args.UNet_batch_size,
                                              shuffle=True if split == 'train' else False,
                                              num_workers=args.num_workers, )
        dataloader[ split ] = loader
    return dataloader


# # UNet

# In[18]:


def convrelu(in_channels, out_channels, kernel, padding):
    
    return nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                          nn.ReLU(inplace=True),
                        )


# In[19]:


class ResNetUNet50(nn.Module):
    
    def __init__(self, layer1=None, layer2=None, layer3=None, layer4=None):
        super().__init__()

        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        #self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(
              nn.Conv2d(12, 64, 7, stride = 2, padding = 3, bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
             )
        
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = layer1
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = layer2
        self.layer2_1x1 = convrelu(512, 512, 1, 0)
        self.layer3 = layer3
        self.layer3_1x1 = convrelu(1024, 1024, 1, 0)
        self.layer4 = layer4
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(1024 + 2048, 2048, 3, 1)
        self.conv_up2 = convrelu(512 + 2048, 1024, 3, 1)
        self.conv_up1 = convrelu(256 + 1024, 512, 3, 1)
        self.conv_up0 = convrelu(64 + 512, 256, 3, 1)

        self.conv_original_size0 = convrelu(12, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 256, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, input): #  torch.rand((1,3,512,512))
        # same as double conv in UNet 
        x_original = self.conv_original_size0(input) # torch.Size([1, 64, 512, 512])
        x_original = self.conv_original_size1(x_original) # torch.Size([1, 64, 512, 512])

        layer0 = self.layer0(input)  # layer0:  torch.Size([1, 64, 256, 256])
        layer1 = self.layer1(layer0) # layer1:  torch.Size([1, 256, 128, 128])
        layer2 = self.layer2(layer1) # layer2:  torch.Size([1, 512, 64, 64])
        layer3 = self.layer3(layer2) # layer3:  torch.Size([1, 1024, 32, 32])
        layer4 = self.layer4(layer3) # layer4:  torch.Size([1, 2048, 16, 16])
        
        
        layer4 = self.layer4_1x1(layer4) # torch.Size([1, 2048, 16, 16])
        
        x = self.upsample(layer4) #torch.Size([1, 2048, 32, 32])
        layer3 = self.layer3_1x1(layer3) # torch.Size([1, 1024, 32, 32])
        x = torch.cat([x, layer3], dim=1) # cat x:  torch.Size([1, 3072, 32, 32])
        x = self.conv_up3(x) # torch.Size([1, 2048, 32, 32])
        
        
        x = self.upsample(x) # torch.Size([1, 2048, 64, 64])
        layer2 = self.layer2_1x1(layer2) # torch.Size([1, 512, 64, 64])
        x = torch.cat([x, layer2], dim=1) #  cat x:  torch.Size([1, 2560, 64, 64])
        x = self.conv_up2(x) # torch.Size([1, 1024, 64, 64])
        
        x = self.upsample(x) # torch.Size([1, 1024, 128, 128])
        layer1 = self.layer1_1x1(layer1) # torch.Size([1, 256, 128, 128])
        x = torch.cat([x, layer1], dim=1) # cat x:  torch.Size([1, 1280, 128, 128])
        x = self.conv_up1(x) # torch.Size([1, 512, 128, 128])
        
        
        x = self.upsample(x) # torch.Size([1, 512, 256, 256])
        layer0 = self.layer0_1x1(layer0) # torch.Size([1, 64, 256, 256])
        x = torch.cat([x, layer0], dim=1) # cat x:  torch.Size([1, 576, 256, 256]) 
        x = self.conv_up0(x) # torch.Size([1, 256, 256, 256])
        
        
        x = self.upsample(x) # torch.Size([1, 256, 512, 512])
        x = torch.cat([x, x_original], dim=1) # cat x:  torch.Size([1, 320, 512, 512])
        x = self.conv_original_size2(x) # torch.Size([1, 64, 512, 512])
        
        out = self.conv_last(x) # torch.Size([1, 1, 512, 512])
        
        return out


# # Resnet50

# In[20]:


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 256, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.Avgpooling = nn.Sequential(self.base_layers[8])
        
        self.mlp = torch.nn.Sequential( torch.nn.Linear(2048, 1), )# only one linear layer on top
        
    def forward(self, X):
        layer0 = self.layer0(X)  # layer0:  torch.Size([1, 64, 256, 256])
        layer1 = self.layer1(layer0) # layer1:  torch.Size([1, 256, 128, 128])
        layer2 = self.layer2(layer1) # layer2:  torch.Size([1, 512, 64, 64])
        layer3 = self.layer3(layer2) # layer3:  torch.Size([1, 1024, 32, 32])
        layer4 = self.layer4(layer3) # layer4:  torch.Size([1, 2048, 16, 16])
        out    = self.Avgpooling(layer4)
        
        out = torch.squeeze(out) 
        if X.size(0) == 1: # torch.Size([1, 2048, 1, 1]) => torch.Size([2048])
            out = torch.unsqueeze(out, 0) # torch.Size([1, 2048])
        out = self.mlp(out)
        return nn.Sigmoid()(out).type(torch.float64)


# # SIMCLR

# In[21]:


class SIMCLR(nn.Module):
    def __init__(self, feature_dim=128, layer0=None, layer1=None, layer2=None, layer3=None, layer4=None, Avgpooling = None):
        super(SIMCLR, self).__init__()

        self.layer0 = layer0 
        self.layer1 = layer1 # size=(N, 256, x.H/4, x.W/4)
        self.layer2 = layer2  # size=(N, 512, x.H/8, x.W/8)
        self.layer3 = layer3   # size=(N, 1024, x.H/16, x.W/16)
        self.layer4 = layer4   # size=(N, 2048, x.H/32, x.W/32)
        self.Avgpooling = Avgpooling
        
        #self.f = base_SimCLR
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        #x = self.f(x)
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0) # layer1:  torch.Size([1, 256, 128, 128])
        layer2 = self.layer2(layer1) # layer2:  torch.Size([1, 512, 64, 64])
        layer3 = self.layer3(layer2) # layer3:  torch.Size([1, 1024, 32, 32])
        layer4 = self.layer4(layer3) # layer4:  torch.Size([1, 2048, 16, 16])
        Avgp   = self.Avgpooling(layer4)
        
        feature = torch.flatten(Avgp, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# # init model

# In[22]:


def init_models(args):
    
    model = Model()

    #unet
    Unet = ResNetUNet50(model.layer1, model.layer2, model.layer3, model.layer4)
    
    #SIMCLR
    SimCLR = SIMCLR(feature_dim, model.layer0, model.layer1, model.layer2, model.layer3, model.layer4, model.Avgpooling)

    Unet.to( args.device )
    model.to( args.device )
    SimCLR.to( args.device )

    return model, Unet, SimCLR


# # Explainer

# In[23]:


class Explainer:
    def __init__(self, model):

        if isinstance(model, nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
            
        self.method = None

    def explain(self, img, target, **kwargs):
        
        target = torch.tensor([[target]])
        heatmap = self.method.attribute(img, target=target, **kwargs)
        
        return heatmap.squeeze().mean(dim=0)


# In[24]:


class GradCamExplainer(Explainer):
    def __init__(self, model, conv_layer):
        super().__init__(model)
        self.method = cattr.LayerGradCam(self.model, conv_layer)

    def explain(self, img, target):
        H, W = img.shape[2], img.shape[3]

        assert img.shape[1:] == torch.Size([3, 512, 512]), "Image dimensions or channels do not match."
        assert target.shape == (img.shape[0],) 
        assert target.device == img.device 
        assert torch.all(target == 0)
        assert isinstance(self.model, nn.DataParallel) == False
        
        
        heatmap = self.method.attribute(img, target, relu_attributions=True)
        
        return cattr.LayerAttribution.interpolate(heatmap, (H, W)).squeeze()


# In[25]:


class DataLoader:
    def __init__(self, input_dir, args):
        
        self.input_dir = input_dir
        self.args = args

    def load_dataframe(self, filename):
        filepath = os.path.join(self.input_dir, filename)
        df = pd.read_csv(filepath)
        
        return df[ df[self.args.attribute_name] == 1 ]


# In[26]:


class ModelLoader:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def load_model(self):
        model = self.model.to(self.args.device)
        model = nn.DataParallel(model)
        
        model_path = os.path.join(self.args.record_dir, self.args.save_name + '_model.pt')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        
        return model


# In[27]:


class H5pyGenerator:
    def __init__(self, args, img_dir, mask_dir, generate_Heatmap_dir, epoch ):
        
        self.args = args
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.generate_Heatmap_dir = generate_Heatmap_dir
 
        self.generate_Heatmap_train_dir = self.generate_Heatmap_dir + f'/train_{epoch}'
        self.generate_Heatmap_val_dir = self.generate_Heatmap_dir + f'/val_{epoch}'

        self.IMG_FILE_REGEX = r'ISIC_(\d+).jpg'
        self.MASK_FILE_REGEX = r'ISIC_(\d+)_attribute_(.*).png'
        self.EXPLAINABLE_FILE_REGEX = r'ISIC_(\d+)_(\d+).png'

        self.ATTR_TO_INDEX = { 'globules': 0,
                               'milia_like_cyst': 1,
                               'negative_network': 2,
                               'pigment_network': 3,
                               'streaks': 4,
                             }

        self.LAYERS_TO_INDEX = {f'{i}{j}': i*3+j - 12 for i in range(4, 8) for j in range(3)}

    def generate_h5py(self, df, output_filename, heatmap_dir ):
        
        output_path = os.path.join( self.generate_Heatmap_dir, output_filename)

        df_sample = df.sample(frac=1).reset_index(drop=True)
        
        with h5py.File(output_path, "w") as f:
            count = len(df_sample)
            images = f.create_dataset('images', (count, 3, 512, 512), dtype=np.uint8)
            image_ids = f.create_dataset('image_ids', (count,), dtype=int)
            img_id_to_h5idx = {}
            
            for i, row in df_sample.iterrows():
                img_file = f"{row['image']}.jpg"
                img = Image.open(os.path.join(self.img_dir, img_file)).resize((512, 512))
                images[i] = np.array(img).transpose(2, 0, 1)
                img_id = int(re.search( self.IMG_FILE_REGEX, img_file ).group(1))
                image_ids[i] = img_id
                img_id_to_h5idx[img_id] = i
            
            masks = f.create_dataset('masks', (count, 512, 512), dtype=np.uint8)
            attr_labels = f.create_dataset('labels', (count,), dtype=np.uint8)
            for mask_file in os.listdir(self.mask_dir):
                if not mask_file.endswith(f'_attribute_{self.args.attribute_name}.png'):
                    continue
                m = re.search( self.MASK_FILE_REGEX, mask_file )
                if len(df[df['image'].str.contains(m.group(1))]) == 0:
                    continue
                    
                mask = Image.open(os.path.join(self.mask_dir, mask_file)).resize((512, 512))
                img_id, attr_id = int(m.group(1)), ATTR_TO_INDEX[m.group(2)]
                h5idx = img_id_to_h5idx[img_id]

                mask = np.array(mask)
                masks[h5idx] = mask
                
                if mask.max() > 0:
                    attr_labels[h5idx] = 1
            
            heatmaps = f.create_dataset('heatmaps', (count, 12, 512, 512), dtype=np.uint8)
            for heatmap_filename in os.listdir( heatmap_dir ):
                if not heatmap_filename.endswith('.png'):
                    continue
                m = re.search( self.EXPLAINABLE_FILE_REGEX, heatmap_filename)
                if len(df[df['image'].str.contains(m.group(1))]) == 0:
                    continue
                heatmap = Image.open(os.path.join( heatmap_dir , heatmap_filename )).resize((512, 512))
                img_id, layer_id = int(m.group(1)), self.LAYERS_TO_INDEX[str(m.group(2))]
                h5idx = img_id_to_h5idx[img_id]
                heatmaps[h5idx, layer_id] = np.array(heatmap)


# In[28]:


class HeatmapGenerator:
    
    def __init__(self, model, args):
        
        self.model = model
        self.args = args
        self.attr_name = args.attribute_name


        # because label only have 1; if you have multi labels, please change to 0, 1, 2, 3, 4
        self.ATTR_TO_INDEX = { 'globules': 0,
                               'milia_like_cyst': 0,
                               'negative_network': 0,
                               'pigment_network': 0,
                               'streaks': 0,
                             }
        self.eps = 1e-10
        

    def get_heatmaps(self, images, batch_size=8, threshold = 0.7):

        all_heatmaps = []
        
        with torch.no_grad():
            
            for i in range(0, len(images), batch_size):
                
                batch_images = images[i:i + batch_size].float().to( self.args.device )
                
                targets = torch.full((batch_images.size(0),), self.ATTR_TO_INDEX[self.attr_name], device = self.args.device )
                heatmaps = self.explainer.explain(batch_images, targets)

                assert  heatmaps.shape[1:] == torch.Size([512, 512])
                
                for j in range(heatmaps.size(0)):
                    heatmaps[j] = heatmaps[j] / (heatmaps[j].max() + self.eps)
                    heatmaps[j][heatmaps[j] < threshold ] = 0
                
                all_heatmaps.append(heatmaps * 255)
        
        return torch.cat(all_heatmaps)

    def generate(self, images):
        
        heatmaps = torch.zeros((len(images), 4, 3, 512, 512), dtype=torch.float64)
        
        for i in range(4, 8):
            for j in range(3):
                
                conv_layer = self.model.module.base_layers[i][j]
                self.explainer = GradCamExplainer(self.model, conv_layer)
                heatmaps[:, i - 4, j, :, :] = self.get_heatmaps( images )
        
        return heatmaps


# In[29]:


class Generate_Heatmap_script:
    def __init__(self, args, epoch = 0):
        
        self.args = args
        self.epoch = epoch
        self.data_loader = DataLoader(args.csv_h5_dir, args)
        self.model_loader = ModelLoader(Model(), args)

        self.img_dir = os.path.join( args.dataset, 'ISIC2018_Task1-2_Training_Input')
        self.mask_dir = os.path.join( args.dataset, 'ISIC2018_Task2_Training_GroundTruth_v3')
        self.generate_Heatmap_dir = args.generate_Heatmap_dir

        self.args.output_dir = args.generate_Heatmap_dir + f'/train_{epoch}'
        self.args.output_dir_val = args.generate_Heatmap_dir + f'/val_{epoch}'

    def run(self):
        df_train = self.data_loader.load_dataframe('train_fold_1.csv')
        df_val = self.data_loader.load_dataframe('val_fold_1.csv')
        
        self.images = torch.load(os.path.join(self.args.H5py_PT, f'{args.attribute_shortname}_images.pt'))
        self.images_val = torch.load(os.path.join(self.args.H5py_PT, f'{args.attribute_shortname}_images(val).pt'))

        assert self.images.shape[1:] == torch.Size([3, 512, 512])
        assert self.images_val.shape[1:] == torch.Size([3, 512, 512])
        
        model = self.model_loader.load_model()
        
        heatmap_generator = HeatmapGenerator(model, args)
        heatmaps_train = heatmap_generator.generate(self.images)
        heatmaps_val = heatmap_generator.generate(self.images_val)

        print('start saving training heatmaps')
        self.save_heatmaps(heatmaps_train, df_train, self.args.output_dir)

        print('start saving validation heatmaps')
        self.save_heatmaps(heatmaps_val, df_val, self.args.output_dir_val)

        self.h5py_gen = H5pyGenerator( self.args, self.img_dir, self.mask_dir, self.generate_Heatmap_dir , self.epoch)
        self.h5py_gen.generate_h5py(df_train, 'train_heatmap.h5', self.args.output_dir)
        self.h5py_gen.generate_h5py(df_val, 'val_heatmap.h5', self.args.output_dir_val)

    def save_heatmaps(self, heatmaps, df, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(heatmaps.size(0)):
            for j in range(heatmaps.size(1)):
                for k in range(heatmaps.size(2)):
                    heatmap = heatmaps[i][j][k].detach().cpu().numpy()
                    im = Image.fromarray(heatmap.astype(np.uint8))
                    im.save(os.path.join(output_dir, f"{df.image.tolist()[i]}_{j+4}{k}.png"))


# # Trainer

# In[30]:


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[31]:


class Trainer:
    def __init__(self, resnet, Unet, SimCLR, dataloaders, args):
        self.resnet = resnet
        self.Unet = Unet
        self.SimCLR = SimCLR

        if True:
            self.resnet = nn.DataParallel(self.resnet, device_ids = args.gpu_ids)
            self.Unet = nn.DataParallel(self.Unet,device_ids = args.gpu_ids)
            self.SimCLR = nn.DataParallel(self.SimCLR, device_ids = args.gpu_ids)
        
        self.dataloaders = dataloaders
        self.args = args
        self.device = args.device
        self.report_freq = args.report_freq

        self.logger = logging.getLogger('Trainer')
        self.logger.setLevel(logging.INFO)  # Set to DEBUG if you want to log debug info too
        fh = logging.FileHandler('log.txt')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.set_loss()
        self.set_optimizer()

        self.unsup_stats = {"loss": []}
        self.UNet_train_stats = {"loss": [], "dice": []}
        self.UNet_val_stats = {"loss": [], "dice": []}
        self.resnet_train_stats = {"loss": [], "acc": [], "auc_scores": []}
        self.resnet_val_stats = {"loss": [], "acc": [], "auc_scores": []}

    def Dice(self, preds, masks):
        smooth = 1e-6
        iflat = preds.contiguous().view(-1)
        tflat = masks.contiguous().view(-1)
    
        # Ensure that predictions are probabilities between 0 and 1
        assert torch.all((iflat >= 0) & (iflat <= 1)), "Prediction values should be between 0 and 1."
    
        # Ensure masks are binary (0 or 1)
        assert torch.all((tflat == 0) | (tflat == 1)), "Mask values should be binary (0 or 1)."
    
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat)
        B_sum = torch.sum(tflat)
    
        # Assert sums to avoid division by zero and ensure non-negative denominators
        assert A_sum + B_sum + smooth > 0, "Denominator in Dice calculation should be greater than zero."
    
        # Compute Dice coefficient
        dice_coefficient = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    
        # Check that the Dice coefficient is within the expected range
        assert dice_coefficient >= 0, "Dice coefficient should be non-negative."
        assert dice_coefficient <= 1, "Dice coefficient should not exceed 1."
    
        return dice_coefficient


    def set_loss(self):
        self.BCELoss = nn.BCELoss()
        self.DiceLoss = DiceLoss()

    def set_optimizer(self):
        self.SimCLR_optimizer = optim.Adam(params=self.SimCLR.parameters(), 
                                           lr=self.args.learn_rate, 
                                           weight_decay=self.args.weight_decay)

        self.Unet_optimizer = optim.Adam(params=self.Unet.parameters(), 
                                         lr=self.args.learn_rate, 
                                         weight_decay=self.args.weight_decay)
        
        self.resnet_optimizer = optim.Adam(params=self.resnet.parameters(), 
                                           lr=self.args.learn_rate, 
                                           weight_decay=self.args.weight_decay)

    def save_model(self, epoch):
        save_path = os.path.join(self.args.record_dir, self.args.save_name + '_model.pt')
        torch.save(self.resnet.state_dict(), save_path)

    def get_accuracy(self, pred, labels):
        predicted_labels = (pred > 0.5).int()
        correct_predictions = (predicted_labels == labels).sum().item()
        total_predictions = labels.size(0)
        accuracy = correct_predictions / total_predictions
        return accuracy

    def unsup_train(self, epoch):
        self.SimCLR.train()
        loss_meter = AverageMeter()
        num_steps = len(self.dataloaders['unsup'])

        for batch_id, (img_ids, imgs1, imgs2) in enumerate(self.dataloaders['unsup']):
            imgs1, imgs2 = imgs1.to(self.device), imgs2.to(self.device)
            feature_1, out_1 = self.SimCLR(imgs1)
            feature_2, out_2 = self.SimCLR(imgs2)
            out = torch.cat([out_1, out_2], dim=0)
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.unsup_batch_size, device=sim_matrix.device)).bool()
            sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.unsup_batch_size, -1)

            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.args.temperature)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            loss.backward()
            self.SimCLR_optimizer.step()
            self.SimCLR_optimizer.zero_grad()

            loss_meter.update(loss.item(), self.args.unsup_batch_size)

            if (batch_id+1) % self.report_freq == 0:
                print('unsup', 
                      f'Epoch: {epoch} ',
                      f'Step: {batch_id}/{num_steps}', 
                      f'Loss: {loss_meter.avg}')

                self.logger.info(f'unsup, Epoch: {epoch} , Step: {batch_id}/{num_steps}, Loss: {loss_meter.avg}')

        self.unsup_stats["loss"].append(loss_meter.avg)

    def train_or_val_ResNet(self, epoch, train=True):
        model = self.resnet if train else self.resnet.module
        dataloader = self.dataloaders['train'] if train else self.dataloaders['val']
        optimizer = self.resnet_optimizer if train else None

        model.train() if train else model.eval()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        num_steps = len(dataloader)

        all_preds, all_labels = [], []
        for batch_id, (img_ids, imgs, labels, _) in enumerate(dataloader):
            if train: optimizer.zero_grad()
            batch_size = len(imgs)
            
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            preds = model(imgs)
            loss = self.BCELoss(preds, labels)
            
            if train:
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item(), batch_size)
            preds, labels = preds.detach().cpu(), labels.cpu()
            acc = self.get_accuracy(preds, labels)
            
            acc_meter.update(acc, batch_size)
            all_preds.append(preds)
            all_labels.append(labels)

            if (batch_id+1) % self.report_freq == 0:
                phase = 'ResNet-TRAIN' if train else 'ResNet-VAL'
                print(f'{phase} Epoch: {epoch} Step: {batch_id}/{num_steps} Loss: {loss_meter.avg:.4f} Accuracy: {acc_meter.avg:.4f}')
                self.logger.info(f'{phase} Epoch: {epoch} Step: {batch_id}/{num_steps} Loss: {loss_meter.avg:.4f} Accuracy: {acc_meter.avg:.4f}')

        stats = {"loss": loss_meter.avg, "acc": acc_meter.avg}
        all_labels = torch.cat(all_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        auc_scores = roc_auc_score(all_labels.numpy(), all_preds.numpy(), average=None)
        stats["auc_scores"] = auc_scores
        return stats

    def train_or_val_UNet(self, epoch, train=True):
        
        model = self.Unet if train else self.Unet.module
        dataloader = self.Heatmaps_dataloader['train'] if train else self.Heatmaps_dataloader['val']
        optimizer = self.Unet_optimizer if train else None

        model.train() if train else model.eval()

        loss_meter = AverageMeter()
        dice_meter = AverageMeter()

        num_steps = len(dataloader)

        all_preds, all_masks = [], []
        start = time.time()

        for batch_id, (img_id, img, labels, masks, heatmaps) in enumerate(dataloader):

            assert masks.shape[1:] == torch.Size([512, 512])
            assert heatmaps.shape[1:] == torch.Size([12, 512, 512])

            if train: optimizer.zero_grad()

            batch_size = len(heatmaps)
            heatmaps, masks = heatmaps.to(self.device), masks.to(self.device)
            masks_copy = masks.clone()
            masks_copy[masks > 1.0] = 1.0

            pred = model(heatmaps).squeeze(1)

            #""""""Ensure predictions are probabilities""""""""""
            pred = torch.sigmoid(pred)  
            assert pred.min() >= 0 and pred.max() <= 1, "Predicted values should be between 0 and 1 after sigmoid activation."
            assert torch.all((masks_copy == 0) | (masks_copy == 1)), "All mask values should be either 0 or 1."
            #""""""""""""""""""""""""""""""""""""""""""""""""""""
            
            loss = self.DiceLoss(pred, masks_copy)
            
            if train:
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item(), batch_size)
            pred, masks_copy = pred.detach().cpu(), masks_copy.cpu()
            dice = self.Dice(pred, masks_copy)

            
            dice_meter.update(dice, batch_size)
            all_preds.append(pred)
            all_masks.append(masks_copy)

            if (batch_id + 1) % self.report_freq == 0:
                phase = 'UNet-TRAIN' if train else 'UNet-VAL'
                print(f'{phase}',
                      f'Epoch: {epoch}',
                      f'Step: {batch_id}/{num_steps}',
                      f'Loss: {loss_meter.avg:.4f}',
                      f'Dice: {dice_meter.avg:.4f}',
                     )

                self.logger.info( f'{phase}, Epoch: {epoch}, Step: {batch_id}/{num_steps}, Loss: {loss_meter.avg:.4f}, Dice: {dice_meter.avg:.4f}' )


        stats = {"loss": loss_meter.avg, "dice": dice_meter.avg}
        all_masks = torch.cat(all_masks, dim=0)
        all_preds = torch.cat(all_preds, dim=0)

        return stats

    def train(self):
        best_loss = float('inf')
        best_model = None
        best_epoch = 0

        for epoch in range(self.args.num_epochs):
            
            self.unsup_train(epoch)

            resnet_train_stats = self.train_or_val_ResNet(epoch, train=True)
            resnet_val_stats = self.train_or_val_ResNet(epoch, train=False)

            self.save_model(epoch)

            GHS = Generate_Heatmap_script(args, epoch)
            GHS.run()

            self.Heatmaps_dataloader = get_heatmap_dataloader( args )

            UNet_train_stats = self.train_or_val_UNet(epoch, train=True)
            UNet_val_stats = self.train_or_val_UNet(epoch, train=False)

            # Save the statistics
            self.resnet_train_stats["loss"].append(resnet_train_stats["loss"])
            self.resnet_train_stats["acc"].append(resnet_train_stats["acc"])
            self.resnet_train_stats["auc_scores"].append(resnet_train_stats["auc_scores"])

            self.resnet_val_stats["loss"].append(resnet_val_stats["loss"])
            self.resnet_val_stats["acc"].append(resnet_val_stats["acc"])
            self.resnet_val_stats["auc_scores"].append(resnet_val_stats["auc_scores"])

            self.UNet_train_stats["loss"].append(UNet_train_stats["loss"])
            self.UNet_train_stats["dice"].append(UNet_train_stats["dice"])

            self.UNet_val_stats["loss"].append(UNet_val_stats["loss"])
            self.UNet_val_stats["dice"].append(UNet_val_stats["dice"])

            self.logger.info(f'Epoch {epoch} ResNet Train Loss: {resnet_train_stats["loss"]:.4f} Train Accuracy: {resnet_train_stats["acc"]:.4f} Train AUC: {resnet_train_stats["auc_scores"]}')
            self.logger.info(f'Epoch {epoch} ResNet Val Loss: {resnet_val_stats["loss"]:.4f} Val Accuracy: {resnet_val_stats["acc"]:.4f} Val AUC: {resnet_val_stats["auc_scores"]}')
            self.logger.info(f'Epoch {epoch} UNet Train Loss: {UNet_train_stats["loss"]:.4f} Train Dice: {UNet_train_stats["dice"]:.4f}')
            self.logger.info(f'Epoch {epoch} UNet Val Loss: {UNet_val_stats["loss"]:.4f} Val Dice: {UNet_val_stats["dice"]:.4f}')

            if resnet_val_stats["loss"] < best_loss:
                best_loss = resnet_val_stats["loss"]
                best_model = self.resnet.module.state_dict()
                best_epoch = epoch

            
            if os.path.exists( GHS.args.output_dir ):
                shutil.rmtree( GHS.args.output_dir )
                self.logger.info(f"Successfully removed {GHS.args.output_dir}")
            else:
                self.logger.info(f"{GHS.args.output_dir} does not exist!")


            if os.path.exists( GHS.args.output_dir_val ):
                shutil.rmtree( GHS.args.output_dir_val )
                self.logger.info(f"Successfully removed {GHS.args.output_dir_val}")
            else:
                self.logger.info(f"{GHS.args.output_dir_val} does not exist!")


            if best_model:
                save_path = os.path.join(self.args.record_dir, self.args.save_name + '_best_model.pt')
                torch.save(best_model, save_path)
                self.logger.info(f'Saved best model from epoch {best_epoch} with validation loss {best_loss:.4f}')
            else:
                self.logger.info(f'Not Best model')


# In[32]:


resnet, Unet, SimCLR = init_models(args)


# In[33]:


# dataloader
num_classes = args.num_classes
dataloaders = get_dataloader( args )


# In[34]:


trainer =   Trainer(resnet, 
                    Unet, 
                    SimCLR, 
                    dataloaders, 
                    args,
                   )


# In[35]:


trainer.train()

