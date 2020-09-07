# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:29:14 2020

@author: REZA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import csv

#%%
Alldata=[]
subjects = 20;

print('[info] reading data')
for i in range(1,65):
        
        Dataset = 'data/20-SubjectsDataset_' + str(i) + '.mat';
        Dataset = loadmat(Dataset);
        Dataset=Dataset['Dataset']   
        Dataset = Dataset.reshape(subjects*84, 640);
        row, column = Dataset.shape
        Dataset = Dataset.T.reshape(1, row *column);
        Alldata.append(Dataset)

print('[info] All the trials extracted and stacked each other. We have 64 lectrode')

#%%Normalize
xx= np.array(Alldata)
NormalizedAll = xx - xx.min();
NormalizedAll = NormalizedAll / xx.max();
NormalizedAll=NormalizedAll.reshape(64,1075200)
print('[info]  Normalizezed')

#%%Covariance
print('[info] Calculating Covariance matrix')
covariance_matrix = np.cov(NormalizedAll);
print('[info] covariance of Normalized/Standardize data is calculated')
np.savetxt("pythondata/foo.csv", covariance_matrix)


plt.style.use('seaborn-poster')
plt.imshow(covariance_matrix,extent=[0,64, 0, 64],cmap='viridis')

#%% Pearson matrix and its ABS
print('[info] Calculating Pearson matrix')
Pearson_matrix= np.corrcoef(NormalizedAll)
np.savetxt("pythondata/Pearson_matrix.csv", Pearson_matrix)
print('[info] Pearson matrix of Normalized/Standardize data is calculated')

print('[info] Calculating Absolute Pearson matrix')
Absolute_Pearson_matrix = abs(Pearson_matrix);
np.savetxt("pythondata/Absolute_Pearson_matrix.csv", Absolute_Pearson_matrix)
print('[info] Absolute Pearson matrix Calculated')

plt.figure()
plt.imshow(Pearson_matrix,extent=[0,64, 0, 64],cmap='viridis')

plt.figure()
plt.imshow(Absolute_Pearson_matrix,extent=[0,64, 0, 64],cmap='viridis')

#%% Adjacency Matrix
print('[info] Calculating Adjacency Matrix')
Eye_Matrix = np.eye(64, 64);
Adjacency_Matrix = Absolute_Pearson_matrix - Eye_Matrix;
np.savetxt("pythondata/Adjacency_Matrix.csv", Adjacency_Matrix)
print('[info] Adjacency Matrix is Calculated')

plt.figure()
plt.imshow(Adjacency_Matrix,extent=[0,64,0, 64],cmap='viridis')



#%% Degree Matrix
print('[info] Calculating Degree Matrix')
diagonal_vector = np.sum(Adjacency_Matrix,axis=1)
Degree_Matrix = np.diag(diagonal_vector)
np.savetxt("pythondata/Degree_Matrix.csv", Degree_Matrix)
print('[info] Degree Matrix Calculated')

plt.figure()
plt.imshow(Degree_Matrix,extent=[0,64,0, 64],cmap='viridis')

#%% Laplacian Matrix

print('[info] Calculating Laplacian Matrix')
Laplacian_Matrix = Degree_Matrix - Adjacency_Matrix;
np.savetxt("pythondata/Laplacian_Matrix.csv", Laplacian_Matrix)
print('[info] Laplacian Matrix Calculated ')

plt.figure()
plt.imshow(Laplacian_Matrix,extent=[0,64,0, 64],cmap='viridis')

#%% Create Labels

print('[info] Creating Label matrix ')

Labels = 'data/20-SubjectsLabels_1.mat';
Labels = loadmat(Labels);
Labels = Labels['Labels']  
Labels = Labels.reshape(subjects*84, 4);
row, column = Labels.shape

Labels = np.argwhere(Labels).T[1]
Labels = Labels.reshape(1680,1)

#%%Extend_Labels

extended=[]
for i in range (640):
    extended.append(Labels)
extended = np.array(extended)
extended = extended.reshape(640,1680)
extended = extended.T
Labels = extended
np.savetxt("pythondata/extended.csv", extended)

row, column = Labels.shape
Labels = Labels.reshape(1,row*column)
Labels=Labels.T
print('[info] Labels are prepared ')

#%% 
# data: 1075200x64
# labels: 1075200x1
print('[info] Train/Test split and ready to run! ')


NormalizedAll= NormalizedAll.T
DATA_ALL = np.append(NormalizedAll,Labels,axis=1)
rowrank =np.random.permutation(1075200)


All_of_Dataset = DATA_ALL[rowrank, :]

row=1075200
#%%
tt=int(np.fix(row/10*9))

training_set   = All_of_Dataset[0:tt , 0:63];
training_label = All_of_Dataset[0:tt , 64];

test_set       = All_of_Dataset[tt:, 0:63];
test_label     = All_of_Dataset[tt:, 64];

np.save("pythondata/training_set", training_set)
np.save("pythondata/training_label", training_label)
np.save("pythondata/test_set", test_set)
np.save("pythondata/test_label", test_label)

print('[info] Everything is ready now! ')
















