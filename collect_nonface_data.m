function [train_nonface_data, train_nonface_label, test_nonface_data1, test_nonface_label1, test_nonface_data2, test_nonface_label2] = collect_nonface_data(cifar10_dir, id, varargin)

resize = false;
if nargin > 2
    resize = varargin{1};
    im_shape = varargin{2};
end
%% read all training data
num_batches = 5;
labels = [];
data = [];
for i=1:num_batches
    cifar_batch = load(fullfile(cifar10_dir, sprintf('data_batch_%d.mat', i)));
    labels = cat(1, labels, cifar_batch.labels);
    data = cat(1, data, cifar_batch.data);
end
labels = labels';     % 1 x 50000
data = double(data'); % 3072 x 50000
% collect training data
train_nonface_data = [];
train_nonface_label = [];
test_nonface_data1 = [];
test_nonface_label1 = [];
for i=1:10
    inds = find(labels == i-1); % original class id starts from 0
    inds = inds(randperm(length(inds))); % shuffle the indices of labels for class i
    
    % train
    inds_train = inds(1:28); 
    data_train = data(:, inds_train);  % 3072 x 28
%     data_train = reshape(data_train, [32, 32, 3, 28]);
%     data_train = permute(data_train, [2, 1, 3, 4]);  % 32 x 32 x 3 x 28
%     data_train = squeeze(mean(data_train, 3));  % 32 x 32 x 28 transform into gray-scale images
    data_train = 0.229 * data_train(1:1024, :) + 0.587 * data_train(1025:2048, :) + 0.114 * data_train(2049:3072, :); % 1024 x 28
    if resize
        data_train_resize = zeros(im_shape(1)*im_shape(2), 28);
        for j=1:28
            im_resize = imresize(reshape(data_train(:, j), [32, 32]), [im_shape(1), im_shape(2)]);
            data_train_resize(:, j) = reshape(im_resize, [im_shape(1)*im_shape(2), 1]);
        end
        data_train = data_train_resize;
    end
    label_train = ones(1, 28) * id;    % 1 x 28
    train_nonface_data = cat(2, train_nonface_data, data_train);    % 1024 x 280
    train_nonface_label = cat(2, train_nonface_label, label_train); % 1 x 280
    
    % test part1
    inds_test = inds(29:35); 
    data_test = data(:, inds_test);  % 3072 x 7
    data_test = 0.229 * data_test(1:1024, :) + 0.587 * data_test(1025:2048, :) + 0.114 * data_test(2049:3072, :); % 1024 x 7
    if resize
        data_test_resize = zeros(im_shape(1)*im_shape(2), 7);
        for j=1:7
            im_resize = imresize(reshape(data_test(:, j), [32, 32]), [im_shape(1), im_shape(2)]);
            data_test_resize(:, j) = reshape(im_resize, [im_shape(1)*im_shape(2), 1]);
        end
        data_test = data_test_resize;
    end
    label_test = ones(1, 7) * id;    % 1 x 7
    test_nonface_data1 = cat(2, test_nonface_data1, data_test);    % 1024 x 70
    test_nonface_label1 = cat(2, test_nonface_label1, label_test); % 1 x 70
end
clear data labels

%% read the test data from cifar10
cifar_batch = load(fullfile(cifar10_dir, 'test_batch.mat'));
labels = cifar_batch.labels';  % 1 x 10000
data = double(cifar_batch.data');      % 3072 x 10000
test_nonface_data2 = [];
test_nonface_label2 = [];
for i=1:10
    inds = find(labels == i-1); % original class id starts from 0
    inds = inds(randperm(length(inds))); % shuffle the indices of labels for class i
    % test part2
    inds_test = inds(1:5); 
    data_test = data(:, inds_test);  % 3072 x 5
    data_test = 0.229 * data_test(1:1024, :) + 0.587 * data_test(1025:2048, :) + 0.114 * data_test(2049:3072, :); % 1024 x 5
    if resize
        data_test_resize = zeros(im_shape(1)*im_shape(2), 5);
        for j=1:5
            im_resize = imresize(reshape(data_test(:, j), [32, 32]), [im_shape(1), im_shape(2)]);
            data_test_resize(:, j) = reshape(im_resize, [im_shape(1)*im_shape(2), 1]);
        end
        data_test = data_test_resize;
    end
    label_test = ones(1, 5) * id;    % 1 x 5
    test_nonface_data2 = cat(2, test_nonface_data2, data_test);    % 1024 x 50
    test_nonface_label2 = cat(2, test_nonface_label2, label_test); % 1 x 50
end

end