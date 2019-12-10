function z = poly_expand(X)
% X: k * N

% expand the squares
z = cat(1, X, X.^2);
% expand the cross-products
for i=1:size(X, 1)-1
    for j=i+1:size(X, 1)
        z = cat(1, z, X(i, :) .* X(j, :));
    end
end