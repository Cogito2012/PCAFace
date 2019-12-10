function [train_data, train_label, test_data1, test_label1, test_data2, test_label2, train_recog_label, test_recog_label] = collect_traindata(dataset_dir, im_shape)

subjs = dir(fullfile(dataset_dir));
subjs=subjs(~ismember({subjs.name},{'.','..'}));
subjs=subjs(cell2mat({subjs.isdir}));

% random shuffle the subjects (35)
nSubj = length(subjs);
subjs = subjs(randperm(nSubj));

subjs_train = subjs(1:35);
subjs_test = subjs(36:end);
train_data = [];
train_label = [];
test_data1 = [];
test_label1 = [];
train_recog_label = [];
test_recog_label = [];
for i=1:length(subjs_train)
    imgs = dir(fullfile(subjs_train(i).folder, subjs_train(i).name, '*.pgm'));
    fprintf('Process training subject: %d\n', i);
    % random shuffle the images (10)
    imgs = imgs(randperm(length(imgs)));
    imgs_train = imgs(1:8);
    imgs_test = imgs(9:end);
    % prepare the training images within the training subjects
    for j=1:length(imgs_train)
        im = imread(fullfile(imgs_train(j).folder, imgs_train(j).name));  % uint8 data
        im_data = double(im);
        im_data = reshape(im_data, [im_shape(1)*im_shape(2), 1]);
        train_data = cat(2, train_data, im_data);  % wh x (35 * 8)
        train_label = cat(2, train_label, 1); % 1 x (35 * 8)
        train_recog_label = cat(2, train_recog_label, i);
    end
    % prepare the testing images within the training subjects
    for j=1:length(imgs_test)
        im = imread(fullfile(imgs_test(j).folder, imgs_test(j).name));  % uint8 data
        im_data = double(im);
        im_data = reshape(im_data, [im_shape(1)*im_shape(2), 1]);
        test_data1 = cat(2, test_data1, im_data);  % wh x (35 * 2)
        test_label1 = cat(2, test_label1, 1); % 1 x (35 * 2)
        test_recog_label = cat(2, test_recog_label, i);
    end
end
% test data from other objects
test_data2 = [];
test_label2 = [];
for i=1:length(subjs_test)
    imgs = dir(fullfile(subjs_test(i).folder, subjs_test(i).name, '*.pgm'));
    fprintf('Process testing subject: %d\n', 35 + i);
    for j=1:length(imgs)
        im = imread(fullfile(imgs(j).folder, imgs(j).name));  % uint8 data
        im_data = double(im);
        im_data = reshape(im_data, [im_shape(1)*im_shape(2), 1]);
        test_data2 = cat(2, test_data2, im_data);  % wh x (35 * 2)
        test_label2 = cat(2, test_label2, 1); % 1 x (35 * 2)
    end
end

end

