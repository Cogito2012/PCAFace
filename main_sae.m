function main_sae()
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
%% Face Recognition
X = cat(2, train_data, train_nonface_data);
y = cat(2, train_label, train_nonface_label);
cls_id = [0, 1];
y = onehot(y, cls_id);  % 2 x 2N
model_file = fullfile(result_dir, 'SAE_recog.mat');
if ~exist(model_file, 'file')
    hiddenSizes = [64; 32; 16];
    autoenc1 = trainAutoencoder(X, hiddenSizes(1));
    features1 = encode(autoenc1,X);
    autoenc2 = trainAutoencoder(features1,hiddenSizes(2));
    features2 = encode(autoenc2,features1);
    autoenc3 = trainAutoencoder(features2,hiddenSizes(3));
    features3 = encode(autoenc3,features2);
    softnet = trainSoftmaxLayer(features3, y, 'LossFunction','crossentropy');
    % construct Stacked AutoEncoders
    SAENet = stack(autoenc1,autoenc2,autoenc3,softnet);
    SAENet = train(SAENet,X,y);
    autoencs = {autoenc1, autoenc2, autoenc3};
    save(model_file, 'SAENet', 'hiddenSizes', 'autoencs');
else
    load(model_file);
end

X = cat(2, test_data1, test_nonface_data1);
y = cat(2, test_label1, test_nonface_label1);

% testing and evaluation
Y_pred = SAENet(X);
Y_test = y;

Y_hat = zeros(1, size(Y_pred, 2));
Y_hat(find(Y_pred(2, :) > Y_pred(1, :))) = 1;
diff = Y_test - Y_hat;
acc = 1 - nnz(diff) / length(diff);
fprintf('Stacked Autoencoder, dimension: 16, face recognition accuracy: %.2f\n', acc);

%%
% latent_feats = encode(autoencs{3}, encode(autoencs{2}, encode(autoencs{1}, test_data1)));
% data_recon = decode(autoenc1, decode(autoenc2, decode(autoenc3, latent_feats)));
% sae_recon_dir = fullfile(result_dir, 'sae_recon_dir');
% if ~exist(sae_recon_dir, 'dir')
%     mkdir(sae_recon_dir);
% end
% im_shape = [112, 92]; 
% for i=1:35
%     im_recon = data_recon(:, 2*i-1);
%     im_recon = (im_recon - min(im_recon)) / (max(im_recon) - min(im_recon));
%     im_recon = reshape(im_recon, im_shape);
%     imwrite(im_recon, fullfile(sae_recon_dir, sprintf('recon%02d.jpg', i)))
% end

%% Face Identification
X = train_data;
y = train_recog_label;
cls_id = unique(train_recog_label); % 1:35
y = onehot(y, cls_id);  % 35 x N
model_file = fullfile(result_dir, 'SAE_identi.mat');
if ~exist(model_file, 'file')
    hiddenSizes = [64; 32; 16];
    autoenc1 = trainAutoencoder(X, hiddenSizes(1));
    features1 = encode(autoenc1,X);
    autoenc2 = trainAutoencoder(features1,hiddenSizes(2));
    features2 = encode(autoenc2,features1);
    autoenc3 = trainAutoencoder(features2,hiddenSizes(3));
    features3 = encode(autoenc3,features2);
    softnet = trainSoftmaxLayer(features3, y, 'LossFunction','crossentropy');
    % construct Stacked AutoEncoders
    SAENet = stack(autoenc1,autoenc2,autoenc3,softnet);
    SAENet = train(SAENet,X,y);
    autoencs = {autoenc1, autoenc2, autoenc3};
    save(model_file, 'SAENet', 'hiddenSizes', 'autoencs');
else
    load(model_file);
end

% testing and evaluation
Y_pred = SAENet(test_data1);
Y_test = test_recog_label;

Y_hat = zeros(1, size(Y_pred, 2));
for i=1:size(Y_pred, 2)
    [~, inds] = max(Y_pred(:, i));
    Y_hat(i) = cls_id(inds);
end
diff = Y_test - Y_hat;
acc = 1 - nnz(diff) / length(diff);
fprintf('Stacked Autoencoder, dimension: 16, face identification accuracy: %.2f\n', acc);
%%
sae_recon_dir = fullfile(result_dir, 'sae_recon_dir');
if ~exist(sae_recon_dir, 'dir')
    mkdir(sae_recon_dir);
end
im_shape = [112, 92]; 
% for i=1:35
i = 1;
    latent_feats = encode(autoencs{3}, encode(autoencs{2}, encode(autoencs{1}, test_data1(:, 2*i-1))));
    im_recon = decode(autoenc1, decode(autoenc2, decode(autoenc3, latent_feats)));
    im_recon = (im_recon - min(im_recon)) / (max(im_recon) - min(im_recon));
    im_recon = reshape(im_recon, im_shape);
    imwrite(im_recon, fullfile(sae_recon_dir, sprintf('recon%02d.jpg', i)))
% end
