clc
clear
close all
%% task 2: Face Recognition with PCA
% input and output directories
dataset_dir = 'att_faces';
result_dir = 'output';
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% collect face data
traindata_file = fullfile(result_dir, 'traindata.mat');
im_shape = [112, 92];
if ~exist(traindata_file, 'file')
    [train_data, train_label, test_data1, test_label1, test_data2, test_label2, train_recog_label, test_recog_label] = collect_traindata(dataset_dir, im_shape);
    save(traindata_file, 'train_data', 'train_label', 'test_data1', 'test_label1', 'test_data2', 'test_label2', 'train_recog_label', 'test_recog_label');
else
    load(traindata_file);
end
% collect non-face data (cifar10)
cifar10_dir = 'cifar10';
neg_id = 0; % we use the class id as 0 for non-face data
nonface_file = fullfile(result_dir, 'nonFaceData.mat');
if ~exist(nonface_file, 'file')
    [train_nonface_data, train_nonface_label, test_nonface_data1, test_nonface_label1, test_nonface_data2, test_nonface_label2] = collect_nonface_data(cifar10_dir, im_shape, neg_id);
    save(nonface_file, 'train_nonface_data', 'train_nonface_label', 'test_nonface_data1', 'test_nonface_label1', 'test_nonface_data2', 'test_nonface_label2');
else
    load(nonface_file);
end
%%  Face Recognition
% Dimension Reduction
type = 'Eigen';
feat_dim = 1024;
[X_train, Y_train] = get_PCA_data(train_data, train_label, train_nonface_data, train_nonface_label, type, feat_dim);

% Linear Regression for Classification
use_onehot = true;
cls_id = [0, 1]; % 1: positive, 0: negative
if use_onehot
    Y_train = onehot(Y_train, cls_id);  % 2 x 2N
    W = Y_train * X_train' * inv(X_train * X_train'); % 2 x K
end
%% predict and evaluate
[X_test, Y_test] = get_PCA_data(test_data1, test_label1, test_nonface_data1, test_nonface_label1, type, feat_dim);
Y_pred = W * X_test;

Y_pred = Y_pred(2, :) > Y_pred(1, :);
% [y_test] = linear_cls(X_train, Y_train, test_data1);

