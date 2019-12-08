function z = poly_expand(X, p)
% X: k * N

if p == 0
    z = X;
else
    z = ones(size(X));
    for i = 1 : p
        z = [z; X.^i];
    end
end