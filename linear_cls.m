function [y_test] = linear_cls(X_train, y_train, X_test)

[~, N] = size(X_train);
nclass = length(unique(y_train));

Y = zeros(nclass, N);
for i = 1:N
    Y(y_train(i), i) = 1;
end

%X = poly_expand(X_train, p);
X = X_train;
W = Y * X' * inv(X * X');

%X_ = poly_expand(X_test, p);
X_ = X_test;
Y_test = W * X_;

[~, ntest] = size(X_test);
y_test = zeros(ntest, 1);
for i = 1:ntest
    [~, idx] = max(Y_test(:, i));
    y_test(i) = idx;
end
y_test = y_test';