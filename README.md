# GCN_for_EEG
Graph Convolutional Networks for 4-class EEG Classification

## The Pure PYTHON interpertation!

### Inspirations :
| Name | Description |
| --- | --- |
| [Michaël Defferrard](https://github.com/mdeff/cnn_graph)| Created the basics of GCN! |
| [Shuyue Jia](https://github.com/SuperBruceJia/EEG-DL) | Created awsome codes for EEG classification|

### What I have done?
Shuyue's work was awsome but preprocessing should be done in MATLAb, which is not available for everyone. So I interperted python version!
I also added other types of GCNs to code  + changed some parts of code

### How to run?
1. Download  - [PhysioNet 4-class EEG](https://physionet.org/content/eegmmidb/1.0.0/) and place it in _01loadData_ folder ( or easily run _downloaddata.py_
2. Run _edfread.py_ . This code will end up in 64 electrode data + 64 Label data. USE PYTHON 2.7
3. It's time to copy the results of previous step into _02Preprocess_ . I placed both MATLAB and PYTHON , But my main intention is having PURE PYTHON environment
So go into _WithPython_ and create a folder called _data_ and place 128 .mat files there. Then run the Code
It results is available in folder _pythondata_ as .csv files
4. create _files_ folder  where the _onEEGcode.py_ is. Then copy CSV files there. Run the _onEEGcode.py_ and ENJOY!

### Dependencies
1. Tensorflow 1.13
2. Numpy
3. Scipy
4. Pandas
5. Pyedflib

### Some Results of Preprocessing

1. Absolute Pearso matrix


<img src="https://github.com/magnumical/GCN_for_EEG/blob/master/Absolute_Pearson_matrix.png" data-canonical-src="https://imreza.ir" width="350" height="350" />



### Classification with basic GCN
<img src="https://github.com/magnumical/GCN_for_EEG/blob/master/acc.png" data-canonical-src="https://imreza.ir" width="450" height="250" />


<img src="https://github.com/magnumical/GCN_for_EEG/blob/master/logom.png" data-canonical-src="https://imreza.ir" width="300" height="100" />

