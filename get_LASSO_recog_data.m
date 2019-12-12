function [X, Y, P] = get_LASSO_recog_data(face_data, face_label, expand, P, varargin)
feat_dim = [];
if nargin > 6
    feat_dim = varargin{1};
end

if isempty(P)
    if isempty(feat_dim)
        % run LASSO algorithm (automatically deterimine the reduced dimension)
        [P, S] = lasso(face_data',face_label'); % D1 x K (K<<D1)
    else
        % run LASSO algorithm (specify the reduced dimension)
        [P, S] = lasso(face_data',face_label', 'DFmax', feat_dim); % D1 x K (K<<D1)
    end
    P = P(:, 1:feat_dim); % D1 x K (K<<D1)
end
X = P' * face_data;  % K x 2N
if expand
    X = poly_expand(X);
end
Y = face_label;   % 1 x 2N
