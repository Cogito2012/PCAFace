clc
clear
close all
%% load dataset
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
X = cat(2, train_data, train_nonface_data);
y = cat(2, train_label, train_nonface_label);
model_file = fullfile(result_dir, 'SAE.mat');
if ~exist(model_file, 'file')
    hiddenSizes = [64; 32; 16];
    autoenc1 = trainAutoencoder(X, hiddenSize(1));
    features1 = encode(autoenc1,X);
    hiddenSize = 32;
    autoenc2 = trainAutoencoder(features1,hiddenSize(2));
    features2 = encode(autoenc2,features1);
    hiddenSize = 16;
    autoenc3 = trainAutoencoder(features2,hiddenSize(3));
    features3 = encode(autoenc3,features2);
    softnet = trainSoftmaxLayer(features3, y, 'LossFunction','crossentropy');
    % construct Stacked AutoEncoders
    SAENet = stack(autoenc1,autoenc2,autoenc3,softnet);
    SAENet = train(SAENet,X,y);
    save(model_file, 'SAENet', 'hiddenSizes');
else
    load(model_file);
end

X = cat(2, test_data1, test_nonface_data1);
y = cat(2, test_label1, test_nonface_label1);

% testing and evaluation
Y_pred = SAENet(X);
Y_test = y;

Y_hat = double(Y_pred > 0.5);
diff = Y_test - Y_hat;
acc = 1 - nnz(diff) / length(diff);
fprintf('Stacked Autoencoder, dimension: 16, face recognition accuracy: %.2f\n', acc);

