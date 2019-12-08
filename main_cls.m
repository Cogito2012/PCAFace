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
id = 2; % we use the class id as 2 rather than 0 for non-face data
nonface_file = fullfile(result_dir, 'nonFaceData.mat');
if ~exist(nonface_file, 'file')
    [train_nonface_data, train_nonface_label, test_nonface_data, test_nonface_label] = collect_nonface_data(cifar10_dir, im_shape, id);
end
%%  Face Recognition
[y_test] = linear_cls(train_data, train_label, test_data1);

