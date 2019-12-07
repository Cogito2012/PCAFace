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

% pca
%%
type = 'Eigen';
% data = data(:, 1:10);
% pca transformation
meandata = mean(data, 2);
[P, s, per] = facePCA(data, meandata, type);

% select the required number of eigen faces
critera = 0.99;
total_per = cumsum(per);
valid_inds = find(total_per > critera);
feat_dim = min(valid_inds);
% feat_dim = 512;

% dimension reduction
P = P(:, 1:feat_dim); % D x 512
F = P' * data;  % 512 x N
X_recon = P * F + meandata; % D x N

% compute the percentage of variance
per = per(1:feat_dim);
total_per = total_per(1:feat_dim);
figure;
% plot(1:length(per), per, 'go');
% hold on
plot(1:length(total_per), total_per, 'ro');
xlabel('PCA index');
ylabel('Percentage of variance');

saveEigenFace = true;
if saveEigenFace
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
end

% face reconstruction
faces_recon_dir = fullfile(result_dir, 'faces_recon_sub1');
if ~exist(faces_recon_dir, 'dir')
    mkdir(faces_recon_dir);
end
figure;
for i=1:10
%     im_recon = mapminmax(X_recon(:, i), 0, 1);
    im_recon = X_recon(:, i);
    im_recon = (im_recon - min(im_recon)) / (max(im_recon) - min(im_recon));
    im_recon = reshape(im_recon, im_shape);
    imwrite(im_recon, fullfile(faces_recon_dir, sprintf('recon%02d.jpg', i)))
    
    subplot(1,2,1)
    title('Original Face');
    im_origin = reshape(data(:, i), im_shape);
    imshow(im_origin)
    
    subplot(1,2,2)
    title('Reconstructed Face');
    imshow(im_recon)
%     pause;
end

