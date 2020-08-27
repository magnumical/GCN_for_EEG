%% 
clear all;clc;format long

%% Stack ALL trials along each other
Alldata = [];
subjects = 20;
disp('[info] starting to reading data')

for i = 1:64
    Dataset = ['data/20-SubjectsDataset_', num2str(i), '.mat'];
    Dataset = load(Dataset);
    Dataset = Dataset.Dataset;
    %84 trial
    %640 samples ( 1 trial 4 sec * 160 Hz)
    Dataset = reshape(Dataset, subjects*84, 640);
    
    [row, column] = size(Dataset);
    Dataset = reshape(Dataset', 1, row*column);
    Alldata = [Alldata; Dataset];
end
disp('[info] All the trials extracted and stacked each other. We have 64 lectrode')

%% Standardize vs Normalize
disp('[info] starting Standardize/Normalize')

%Standardize
%Alldata = Alldata - mean(Alldata, 1);
%Alldata = Alldata'
%disp('[info]  Standardized')

%Normalize
NormalizedAll = Alldata - min(Alldata(:));
NormalizedAll = NormalizedAll ./ max(NormalizedAll(:));
NormalizedAll=NormalizedAll';
disp('[info]  Normalizezed')
%figure();plot(Alldata(1,1:640))
%% Covariance Matrix
% Actually it gives variance but in 64x64 way!

disp('[info] Calculating Covariance matrix')

covariance_matrix = cov(NormalizedAll);
writematrix(covariance_matrix,'finaldata/covariance_matrix.csv');
disp('[info] covariance of Normalized/Standardize data is calculated')

figure();
imagesc(covariance_matrix)
title('Covariance Matrix of 20 Subjects','FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels');colorbar
%print('Covariance', '-dpng',  '-r600')

%% Pearson matrix and its ABS
disp('[info] Calculating Pearson matrix')
Pearson_matrix = corrcoef(NormalizedAll);
writematrix(Pearson_matrix,'finaldata/Pearson_matrix.csv');
disp('[info] Pearson matrix of Normalized/Standardize data is calculated')

disp('[info] Calculating Absolute Pearson matrix')
Absolute_Pearson_matrix = abs(Pearson_matrix);
writematrix(Absolute_Pearson_matrix,'finaldata/Absolute_Pearson_matrix.csv');
disp('[info] Absolute Pearson matrix Calculated')


figure();
imagesc(Pearson_matrix)
title('Pearson Matrix of 20 Subjects','FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels');colorbar
%print('Pearson_Matrix', '-dpng',  '-r600')

figure();
imagesc(Absolute_Pearson_matrix)
title('Absolute Pearson Matrix of 20 Subjects','FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels');colorbar
%print('Absolute_Pearson_matrix', '-dpng',  '-r600')

%% Adjacency Matrix
disp('[info] Calculating Adjacency Matrix')
Eye_Matrix = eye(64, 64);
Adjacency_Matrix = Absolute_Pearson_matrix - Eye_Matrix;
writematrix(Adjacency_Matrix,'finaldata/Adjacency_Matrix.csv');
disp('[info] Adjacency Matrix is Calculated')

figure();
imagesc(Adjacency_Matrix)
title('Adjacency Matrix of 20 Subjects','FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels');colorbar
%print('Adjacency_matrix', '-dpng',  '-r600')

%% Degree Matrix
disp('[info] Calculating Degree Matrix')
diagonal_vector = sum(Adjacency_Matrix, 2);
Degree_Matrix = diag(diagonal_vector);
writematrix(Degree_Matrix,'finaldata/Degree_Matrix.csv');
disp('[info] Degree Matrix Calculated')

figure();
imagesc(Degree_Matrix)
title('Degree Matrix of 20 Subjects','FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels');colorbar

%% Laplacian Matrix
disp('[info] Calculating Laplacian Matrix')
Laplacian_Matrix = Degree_Matrix - Adjacency_Matrix;
writematrix(Laplacian_Matrix,'finaldata/Laplacian_Matrix.csv');
disp('[info] Laplacian Matrix Calculated ')
figure();

imagesc(Laplacian_Matrix)
title('DLaplacian_Matrix of 20 Subjects','FontWeight', 'bold')
xlabel('Channels'), ylabel('Channels');colorbar
print('Laplacian_Matrix', '-dpng',  '-r600')

%% Create Labels
Labels = load('data/20-SubjectsLabels_1.mat');
Labels = Labels.Labels;
Labels = reshape(Labels, subjects*84, 4);
[row, column] = size(Labels);


New_Labels = [];
parfor i = 1:row
    location = find(Labels(i, :) == 1);
    location = location - 1;
    New_Labels = [New_Labels; location];
end
Labels = New_Labels;

Extend_Labels = [];
parfor i =1:640
    Extend_Labels = [Extend_Labels, Labels];
end
Labels = Extend_Labels;

[row, column] = size(Labels);
%Labels.T = 640x1680
Labels = reshape(Labels', 1, row*column);
Labels = Labels';
disp('Labels')
%%
All_Data = [NormalizedAll, Labels];
rowrank = randperm(size(All_Data, 1));
All_Dataset = All_Data(rowrank, :);
[row, ~] = size(All_Dataset);


training_set   = All_Dataset(1:fix(row/10*9),     1:64);
test_set       = All_Dataset(fix(row/10*9)+1:end, 1:64);
training_label = All_Dataset(1:fix(row/10*9),     end);
test_label     = All_Dataset(fix(row/10*9)+1:end, end);


writematrix(training_set,'finaldata/training_set.csv' );
writematrix(test_set,'finaldata/test_set.csv' );
writematrix(training_label,'finaldata/training_label.csv' );
writematrix(test_label,'finaldata/test_label.csv' );
