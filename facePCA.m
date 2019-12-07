function [U, s] = my_pca(data, meandata, type)

%type 1: SVD
%type 2: eigen

[~, n_sample] = size(data);
data = data - meandata;

if type == 1
    var_data = data * data' ./ (n_sample - 1);
    [U, D] = eig(var_data);
    s = diag(D);
    
    [s, idx] = sort(s, 'descend');
    U = U(:, idx);
else
    data_n = data / sqrt(n_sample - 1);
    [U, S, ~] = svd(data_n);
    s = diag(S); % s is already sorted
end
