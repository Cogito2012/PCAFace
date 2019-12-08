function [train_nonface_data, train_nonface_label, test_nonface_data1, test_nonface_label1, test_nonface_data2, test_nonface_label2] = collect_nonface_data(cifar10_dir, im_shape, id)

%% read all training data
num_batches = 5;
labels = [];
data = [];
for i=1:num_batches
    cifar_batch = load(fullfile(cifar10_dir, sprintf('data_batch_%d.mat', i)));
    labels = cat(1, labels, cifar_batch.labels);
    data = cat(1, data, cifar_batch.data);
end
labels = labels';  % 1 x 50000
data = data';      % 3072 x 50000
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
    data_train = reshape(data_train, [32, 32, 3, 28]);
    data_train = permute(data_train, [2, 1, 3, 4]);  % 32 x 32 x 3 x 28
    data_train = squeeze(mean(data_train, 3));  % 32 x 32 x 28 transform into gray-scale images
    for j=1:28
        % todo: resize image as 112 x 92
        % data_train(:, :, j) = 
    end
    label_train = ones(1, 28) * id;    % 1 x 28
    train_nonface_data = cat(2, train_nonface_data, data_train);    % 3072 x 280
    train_nonface_label = cat(2, train_nonface_label, label_train); % 1 x 280
    % test part1
    inds_test = inds(29:35); 
    data_test = data(:, inds_test);  % 3072 x 7
    label_test = ones(1, 7) * id;    % 1 x 7
    test_nonface_data1 = cat(2, test_nonface_data1, data_test);    % 3072 x 70
    test_nonface_label1 = cat(2, test_nonface_label1, label_test); % 1 x 70
end
clear data labels

%% read the test data from cifar10
cifar_batch = load(fullfile(cifar10_dir, 'test_batch.mat'));
labels = cifar_batch.labels';  % 3072 x 10000
data = cifar_batch.data';      % 1 x 10000
test_nonface_data2 = [];
test_nonface_label2 = [];
for i=1:10
    inds = find(labels == i-1); % original class id starts from 0
    inds = inds(randperm(length(inds))); % shuffle the indices of labels for class i
    % test part2
    inds_test = inds(1:5); 
    data_test = data(:, inds_test);  % 3072 x 5
    label_test = ones(1, 5) * id;    % 1 x 5
    test_nonface_data2 = cat(2, test_nonface_data2, data_test);    % 3072 x 50
    test_nonface_label2 = cat(2, test_nonface_label2, label_test); % 1 x 50
end

end