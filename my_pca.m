function [U, s] = my_pca(data, type)

%type 1: SVD
%type 2: eigen

[~, n_sample] = size(data);
data = data - mean(data, 2);

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
