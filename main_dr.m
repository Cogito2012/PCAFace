function main_dr()
%% task 1: EigenFace with PCA, Face reconstruction
%% load data set
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
type = 'SVD';
% data = data(:, 1:10);
% pca transformation
meandata = mean(data, 2);
[PC, s, percentage] = facePCA(data, meandata, type);

% select the required number of eigen faces
critera = 0.99;
total_per = cumsum(percentage);
valid_inds = find(total_per > critera);
feat_dim = min(valid_inds);
% feat_dim = 512;
fprintf('Required number of eigen faces is: %d\n', feat_dim);

% dimension reduction
P = PC(:, 1:feat_dim); % D x 512
F = P' * data;  % 512 x N
X_recon = P * F + meandata; % D x N

% compute the percentage of variance
per = percentage(1:feat_dim);
total_per = total_per(1:feat_dim);
figure;
fontsize = 18;
% plot(1:length(per), per, 'go');
% hold on
plot(1:length(total_per), total_per, 'ro');
hold on;
plot(1:length(per), per, 'b*');
set(gca,'FontSize',fontsize)
xlabel('PC index', 'fontsize', fontsize);
ylabel('Percentage of variance', 'fontsize', fontsize);
legend('cumulative percentage', 'percentage', 'fontsize', fontsize);

%% Number of faces vs. Criteria
total_per = cumsum(percentage);
for criteria=0.80:0.01:0.99
    valid_inds = find(total_per > critera);
    feat_dim = min(valid_inds);
end


 %%
feat_dims = [32, 64, 128, 256, 512, 1024];
% face reconstruction
faces_recon_dir = fullfile(result_dir, 'faces_recon_sub1_im1_svd');
if ~exist(faces_recon_dir, 'dir')
    mkdir(faces_recon_dir);
end
for i=1:length(feat_dims)
    fprintf('Reconstruct with %d eigen faces.\n', feat_dims(i));
    Pi = PC(:, 1:feat_dims(i)); % D x K
    F = Pi' * data(:, 1);  % K x 1
    im_recon = Pi * F + meandata; % D x N
    im_recon = (im_recon - min(im_recon)) / (max(im_recon) - min(im_recon));
    im_recon = reshape(im_recon, im_shape);
    imwrite(im_recon, fullfile(faces_recon_dir, sprintf('recon_dim%03d.jpg', feat_dims(i))));
end

%%
feat_dim = [2, 4, 8, 16, 32, 64, 128];
reg = [0.89, 0.95, 0.99, 0.97, 0.51, 0.54, 0.56];
ver = [0.24, 0.53, 0.91, 0.99, 0.03, 0, 0.03];

plot(feat_dim, reg, 'r-','LineWidth',2)
hold on
plot(feat_dim, ver, 'g-','LineWidth',2)
set(gca,'FontSize',18)
xlabel('Feature dimension', 'FontSize', 18)
ylabel('Classification accuracy', 'FontSize', 18)
lgd = legend({'Face regconition', 'Face identification'}, 'FontSize', 18);
grid on
%%
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

%% face reconstruction
faces_recon_dir = fullfile(result_dir, 'faces_recon_sub1');
if ~exist(faces_recon_dir, 'dir')
    mkdir(faces_recon_dir);
end
figure;
for i=1:10
    % original faces
    subplot(1,3,1)
    im_origin = reshape(data(:, i), im_shape);
    imshow(im_origin)
    title('Original Face');
    
    % reconstructed faces
    im_recon = X_recon(:, i);
    im_recon = (im_recon - min(im_recon)) / (max(im_recon) - min(im_recon));
    im_recon = reshape(im_recon, im_shape);
    imwrite(im_recon, fullfile(faces_recon_dir, sprintf('recon%02d.jpg', i)))
    subplot(1,3,2)
    imshow(im_recon)
    title('Reconstructed Face');
    
    im_diff = abs(im_origin - im_recon);
    subplot(1,3,3);
    imshow(im_diff)
    title('Difference Face')
%     pause;
end

