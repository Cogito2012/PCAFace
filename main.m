clc
clear
close all
%% task 1: EigenFace with PCA, Face reconstruction
% input and output directories
dataset_dir = './att_faces';
result_dir = 'output';
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% collect face data
facedata_file = fullfile(result_dir, 'face_data.mat');
im_shape = [112, 92];
if ~exist(facedata_file, 'file')
    [data, mean_all] = collect_facedata(dataset_dir, im_shape, result_dir);
    save(facedata_file, 'data');
else
    load(facedata_file);
end

% pca
%%
% data = data(:, 1:10);
% pca transformation
[P, s] = my_pca(data, 1);
% dimension reduction
P = P(:, 1:512); % D x 512
F = P' * data;  % 512 x N
X_recon = P * F; % D x N


% save eigen face
eigenFace_dir = fullfile(result_dir, 'eigenFaces');
if ~exist(eigenFace_dir, 'dir')
    mkdir(eigenFace_dir)
end
for i=1:size(P, 2)
%     EigenFace = mapminmax(reshape(P(:, i), im_shape), 0, 1);
    EigenFace = P(:, i);
    EigenFace = (EigenFace - min(EigenFace)) / (max(EigenFace) - min(EigenFace));
    EigenFace = reshape(EigenFace, im_shape);
    imName = sprintf('EigenFace%03d.jpg', i);
    imwrite(EigenFace, fullfile(eigenFace_dir, imName));
end

% face reconstruction
figure;
for i=1:10
    im_recon = mapminmax(X_recon(:, i), 0, 1);
    im_recon = reshape(im_recon, im_shape);
    imshow(im_recon)
    pause;
end

