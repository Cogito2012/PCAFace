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
resize = true;
nonface_file = fullfile(result_dir, 'nonFaceData.mat');
if resize
    nonface_file = fullfile(result_dir, 'nonFaceData_Resize.mat');
end
if ~exist(nonface_file, 'file')
    [train_nonface_data, train_nonface_label, test_nonface_data1, test_nonface_label1, test_nonface_data2, test_nonface_label2] = ...
        collect_nonface_data(cifar10_dir, neg_id, resize, im_shape);
    save(nonface_file, 'train_nonface_data', 'train_nonface_label', 'test_nonface_data1', 'test_nonface_label1', 'test_nonface_data2', 'test_nonface_label2');
else
    load(nonface_file);
end
%%  Face Recognition
type = 'SVD';
feat_dim = 16;
shared = true; 
expand = true;
cls_id = [0, 1]; % 1: positive, 0: negative

% Dimension Reduction
[X_train, Y_train, P] = get_PCA_cls_data(train_data, train_label, train_nonface_data, train_nonface_label, type, feat_dim, shared, expand, []);

% Linear Regression for Classification
Y_train = onehot(Y_train, cls_id);  % 2 x 2N
W = Y_train * X_train' * inv(X_train * X_train'); % 2 x K

% predict labels
[X_test, Y_test, P] = get_PCA_cls_data(test_data1, test_label1, test_nonface_data1, test_nonface_label1, type, feat_dim, shared, expand, P);
Y_pred = W * X_test;

Y_hat = zeros(1, size(Y_pred, 2));
Y_hat(find(Y_pred(2, :) > Y_pred(1, :))) = 1;
diff = Y_test - Y_hat;
acc = 1 - nnz(diff) / length(diff);
fprintf('DR method: %s, dimension: %d, face recognition accuracy: %.2f\n', type, feat_dim, acc);

%% Face Verification
type = 'SVD';
feat_dim = 16;
expand = true;
use_onehot = true;
cls_id = unique(train_recog_label); % 1:35
% Dimension Reduction
[X_train, Y_train, P] = get_PCA_recog_data(train_data, train_recog_label, type, feat_dim, expand, []);

% Linear Regression for Classification
Y_train = onehot(Y_train, cls_id);  % 2 x 2N
W = Y_train * X_train' * inv(X_train * X_train'); % 2 x K

% predict labels
[X_test, Y_test, P] = get_PCA_recog_data(test_data1, test_recog_label, type, feat_dim, expand, P);
Y_pred = W * X_test;

Y_hat = zeros(1, size(Y_pred, 2));
for i=1:size(Y_pred, 2)
    [~, inds] = max(Y_pred(:, i));
    Y_hat(i) = cls_id(inds);
end
diff = Y_test - Y_hat;
acc = 1 - nnz(diff) / length(diff);
fprintf('DR method: %s, dimension: %d, face verification accuracy: %.2f\n', type, feat_dim, acc);
