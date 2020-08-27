# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:48:58 2020

@author: REZA
"""

import numpy as np
import pandas as pd

####################################
#        train_data:   [N_train X M]
#        train_labels: [N_train X 1]
#        test_data:    [N_test X M]
#        test_labels:  [N_test X 1]
#        (N: number of samples, M: number of features)
####################################

def dataread(DIR):
 
    train_data = pd.read_csv(DIR + 'training_set.csv', header=None)
    train_data = np.array(train_data).astype('float32')

    train_labels = pd.read_csv(DIR + 'training_label.csv', header=None)
    train_labels = np.array(train_labels).astype('float32')

    # Read Testing Data and Labels
    test_data = pd.read_csv(DIR + 'test_set.csv', header=None)
    test_data = np.array(test_data).astype('float32')

    test_labels = pd.read_csv(DIR + 'test_label.csv', header=None)
    test_labels = np.array(test_labels).astype('float32')

    return train_data, train_labels, test_data, test_labels

