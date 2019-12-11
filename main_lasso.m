clc
clear
close all

%% Load Data
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
%%

%%  Face Recognition
feat_dim = [];
expand = true;
cls_id = [0, 1]; % 1: positive, 0: negative

% Dimension Reduction
[X_train, Y_train, P] = get_LASSO_cls_data(train_data, train_label, train_nonface_data, train_nonface_label, expand, [], feat_dim);
feat_dim = size(P, 2);

% Linear Regression for Classification
Y_train = onehot(Y_train, cls_id);  % 2 x 2N
W = Y_train * X_train' * inv(X_train * X_train'); % 2 x K

% predict labels
[X_test, Y_test, P] = get_LASSO_cls_data(test_data1, test_label1, test_nonface_data1, test_nonface_label1, expand, P, feat_dim);
Y_pred = W * X_test;

Y_hat = zeros(1, size(Y_pred, 2));
Y_hat(find(Y_pred(2, :) > Y_pred(1, :))) = 1;
diff = Y_test - Y_hat;
acc = 1 - nnz(diff) / length(diff);
fprintf('Lasso method, dimension: %d, face recognition accuracy: %.2f\n', feat_dim, acc);

%%
% visualize B (sparse coefficients)
nr = floor(size(P, 1)/64);
nc = size(P, 2);
im_P = zeros(nr, nc);
for i=1:64
    im_P(i, :) = mean(P((i-1)*64+1:i*64, :), 1);
    inds = find(abs(im_P(i, :))~=0);
    im_P(i, inds) = 1;
end
imshow(im_P)




