clc
clear
close all
%% task 1: EigenFace with PCA, Face reconstruction
% input and output directories
dataset_dir = 'att_faces';
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
%%
save_dir = fullfile(result_dir, 'images');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
for i=1:10
    im = reshape(data(:, i), [112, 92]);
    imwrite(im, fullfile(save_dir, sprintf('p1_%02d.png',i)));
end
