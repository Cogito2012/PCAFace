function [U, s, per] = facePCA(data, meandata, type)

%type 1: Eigen Deconposition
%type 2: SVD

[~, n_sample] = size(data);
data = data - meandata;

if strcmp(type, 'Eigen')
    var_data = data * data' ./ (n_sample - 1);
    [U, D] = eig(var_data);
    s = diag(D);
    
    [s, idx] = sort(s, 'descend');
    U = U(:, idx);
    
    tv = sum(s);
    per = zeros(1, length(s));
    for i=1:length(s)
        per(i) = s(i) / tv;
    end
elseif strcmp(type, 'SVD')
    data_n = data / sqrt(n_sample - 1);
    [U, S, ~] = svd(data_n);
    s = diag(S); % s is already sorted
    
    tv = sum(s.^2);
    per = zeros(1, length(s));
    for i=1:length(s)
        per(i) = s(i)^2 / tv;
    end
else
    error('Unsupported pca method!')
end
