#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 06:16:51 2019

@author: Kuangen Zhang
"""
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../VisionProcess'))
from FileIO import FileIO
import provider

# load dataset
TRAIN_FILES_VEC = [[],[]]
TEST_FILES_VEC = [[],[]]
for i in range(1):
    path = 'data/extracted_feature'
#    path = 'data/modelnet40_ply_hdf5_2048_0/'
    TRAIN_FILES_VEC[i] = provider.getDataFiles( \
        os.path.join(BASE_DIR, path + '/train_files.txt'))
    TEST_FILES_VEC[i] = provider.getDataFiles(\
        os.path.join(BASE_DIR, path + '/test_files.txt'))

## data are point cloud 
#data = np.array([], dtype=np.float32).reshape(0,2048,3)
#label = np.array([], dtype=np.float32).reshape(0,1)
    
## data are features
data = np.array([], dtype=np.float32).reshape(0,1024) 
label = np.array([], dtype=np.float32).reshape(0)
for i in range(len(TEST_FILES_VEC[0])):
    data_temp, label_temp = FileIO.load_h5(TEST_FILES_VEC[0][i])
    data = np.concatenate((data, data_temp), axis=0)
    label = np.concatenate((label, label_temp), axis=0)

## data are point cloud 
#data = data[:,0:1024,:]
#data = data.reshape((-1,3*1024))

## data are features
data = np.concatenate([data, np.zeros((
                data.shape[0], 3*1024 - data.shape[1]))], axis  = -1)
#%%
## T-SNE to reduce dimension 
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(data)
#%%
# plot T-SNE
label = np.squeeze(label)
plt.figure(figsize=(6, 6))
plt.scatter(tsne[:, 0], tsne[:, 1], c=label,cmap=plt.get_cmap('hsv'))
plt.axis('off')
plt.colorbar()
plt.show()
