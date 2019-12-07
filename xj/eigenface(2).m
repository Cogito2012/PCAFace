function [] = eigenface(root, type)

data = zeros(92 * 112, 400);

for i = 1:40
    for j = 1:10
        path = [root, '/s', num2str(i), '/', num2str(j), '.pgm'];
        raw_img = imread(path);
        raw_img = im2double(raw_img);
        data(:, (i - 1) * 10 + j) = reshape(raw_img, [], 1);
    end
end

[P, s, X_new] = my_pca(data, type);
X_recon = P * X_new;
sample = X_recon(:, 1);
img_min = min(sample);
img_max = max(sample);
img = (sample - img_min) / (img_max - img_min);
img = reshape(img, 112, 92);
imshow(img);