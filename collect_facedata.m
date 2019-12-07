function [data, mean_all] = collect_facedata(dataset_dir, im_shape, result_dir)

subjs = dir(fullfile(dataset_dir));
subjs=subjs(~ismember({subjs.name},{'.','..'}));
subjs=subjs(cell2mat({subjs.isdir}));

nSubj = length(subjs);
mean_all = zeros(im_shape);
data = [];
nImgs = 0;
for i=1:nSubj
    imgs = dir(fullfile(subjs(i).folder, subjs(i).name, '*.pgm'));
    fprintf('Process subject: %d\n', i);
    mean_subj = zeros(im_shape);
    for j=1:length(imgs)
        im = imread(fullfile(imgs(j).folder, imgs(j).name));  % uint8 data
        im_data = double(im) / 255.0;
        mean_all = mean_all + im_data;
        mean_subj = mean_subj + im_data;
        data = cat(2, data, reshape(im_data, [im_shape(1)*im_shape(2), 1]));
        nImgs = nImgs + 1;
    end
    mean_subj = mean_subj / length(imgs);
    %imshow(mean_subj)
    subID = sprintf('mean_sub%02d', i);
    imwrite(mean_subj, fullfile(result_dir, [subID, '.jpg']));
end
mean_all = mean_all / nImgs;
imwrite(mean_all, fullfile(result_dir, 'mean_all.jpg'));

end

