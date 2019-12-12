function [X, Y, P] = get_LASSO_cls_data(face_data, face_label, nonface_data, nonface_label, expand, P, varargin)

feat_dim = [];
if nargin > 6
    feat_dim = varargin{1};
end
data = cat(2, face_data, nonface_data);  % (D1 x 2N)
label = cat(2, face_label, nonface_label);   % 1 x 2N
if isempty(P)
    if isempty(feat_dim)
        % run LASSO algorithm (automatically deterimine the reduced dimension)
        [P, S] = lasso(data',label'); % D1 x K (K<<D1)
    else
        % run LASSO algorithm (specify the reduced dimension)
        [P, S] = lasso(data',label', 'DFmax', feat_dim); % D1 x K (K<<D1)
    end
end
X = P' * data;  % K x 2N
if expand
    X = poly_expand(X);
end
Y = label;
end