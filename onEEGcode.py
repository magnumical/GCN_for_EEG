# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 01:52:33 2020

@author: REZA
"""

from scripts import models, graph, coarsening,GCN_Model,DenseGCN_Modelc
from scripts.dataread import dataread
#%%
import numpy as np
import pandas as pd

from scipy import sparse
from tensorflow.python.framework import ops

import tensorflow as tf
# Clear all the stack and use GPU resources as much as possible
ops.reset_default_graph()
config= tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)


#%%

train_data= np.load('files/training_set.npy').astype('float32')
train_labels= np.load('files/training_label.npy').astype('float32')

test_data= np.load('files/test_set.npy').astype('float32')
test_labels= np.load('files/test_label.npy').astype('float32')

print('==============> Data read!')
  
#%%
test_labels=test_labels.reshape(107520,)
train_labels=train_labels.reshape(967680,)
#train_data=train_data.reshape(64,967680)
#test_data=test_data.reshape(64,107520)

#%%
Adjacency_Matrix = pd.read_csv(DIR+'Adjacency_Matrix.csv', header=None)
Adjacency_Matrix = np.array(Adjacency_Matrix).astype('float32')
Adjacency_Matrix = sparse.csr_matrix(Adjacency_Matrix)
print('==============> Adjancy matrix read!')


graphs, perm = coarsening.coarsen(Adjacency_Matrix, levels=5, self_connections=False)
X_train = coarsening.perm_data(train_data, perm)
X_test  = coarsening.perm_data(test_data,  perm)
print('==============>coarsening done!')


#%%


L = [graph.laplacian(Adjacency_Matrix, normalized=True) for Adjacency_Matrix in graphs]
print('==============>laplacian obtained!')
graph.plot_spectrum(L)

#%% Hyper-parameters
params = dict()
params['dir_name']       = 'folder1'

params['num_epochs']     = 100
params['batch_size']     = 1024
params['eval_frequency'] = 100

# Building blocks.
params['filter'] = 'chebyshev5'
params['brelu']  = 'b2relu'
params['pool']   = 'mpool1'

# Architecture.
params['F'] = [16, 32, 64, 128, 256, 512]  # Number of graph convolutional filters.
params['K'] = [2, 2, 2, 2, 2, 2]           # Polynomial orders.
params['p'] = [2, 2, 2, 2, 2, 2]           # Pooling sizes.
params['M'] = [4]                          # Output dimensionality of fully connected layers.

# Optimization.
params['regularization'] = 0.000001  # L2 regularization
params['dropout']        = 0.50      # Dropout rate
params['learning_rate']  = 0.000001  # Learning rate
params['decay_rate']     = 1         # Learning rate Decay == 1 means no Decay
params['momentum']       = 0         # momentum == 0 means Use Adam Optimizer
params['decay_steps']    = np.shape(train_data)[0] / params['batch_size']
print('==============>parameters  selected!')

#%%
model = models.cgcnn(L, **params)
#model = GCN_Model.cgcnn(L, **params)
#model=DenseGCN_Model.cgcnn(L, **params)

print('==============>model  established!')
accuracy, loss, t_step = model.fit(train_data, train_labels, test_data, test_labels)

print('==============>model  fitted!')



