function [P, s, X_new] = my_pca(data, type)

%type 1: SVD
%type 2: eigen

[~, n_sample] = size(data);
data = data - mean(data, 2) * ones(1, n_sample);

if type == 1
    var_data = data * data' ./ (n_sample - 1);
    [U, D] = eig(var_data);
    s = diag(D);
    
    [s, idx] = sort(s, 'descend');
    U = U(:, idx);
    
    P = U;
else
    data_n = data / sqrt(n_sample - 1);
    [U, S, ~] = svd(data_n);
    
    s = diag(S);
    P = U;
end

P = P(:, 1:512);
X_new = P' * data;