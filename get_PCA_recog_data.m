function [X, Y, P] = get_PCA_recog_data(face_data, face_label, type, feat_dim, expand, P)

if isempty(P)
    [P, ~, ~] = facePCA(face_data, mean(face_data, 2), type);
    P = P(:, 1:feat_dim); % D1 x K (K<<D1)
end
X = P' * face_data;  % K x 2N
if expand
    X = poly_expand(X);
end
Y = face_label;   % 1 x 2N
