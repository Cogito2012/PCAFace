function z = poly_expand(X)
% X: k * N

K = size(X, 1);
z = zeros(K*(K+3)/2, size(X, 2));
z(1:K, :) = X;
% expand the squares
% z = cat(1, X, X.^2);
z(K+1:2*K, :) = X.^2;
% expand the cross-products
n = 1;
for i=1:K-1
    for j=i+1:K
        % z = cat(1, z, X(i, :) .* X(j, :));
        z(2*K+n, :) = X(i, :) .* X(j, :);
        n = n + 1;
    end
end
