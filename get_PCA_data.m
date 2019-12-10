function [X, Y] = get_PCA_data(face_data, face_label, nonface_data, nonface_label, type, feat_dim, shared, expand)

if shared
    data = cat(2, face_data, nonface_data);  % (D1 x 2N)
    [P, s, per] = facePCA(data, mean(data, 2), type);
    P = P(:, 1:feat_dim); % D1 x K (K<<D1)
    X = P' * data;  % K x 2N
    if expand
        X = poly_expand(X);
    end
    Y = cat(2, face_label, nonface_label);   % 1 x 2N
else
    % positive
    [P, s, per] = facePCA(face_data, mean(face_data, 2), type);
    P = P(:, 1:feat_dim); % D1 x K (K<<D1)
    X_pos = P' * face_data;  % K x N
    if expand
        X_pos = poly_expand(X_pos);
    end
    Y_pos = face_label; % 1 x N

    % negative
    [P, s, per] = facePCA(nonface_data, mean(nonface_data, 2), type);
    P = P(:, 1:feat_dim); % D2 x K (K<<D2)
    X_neg = P' * nonface_data;  % K x N
    if expand
        X_neg = poly_expand(X_neg);
    end
    Y_neg = nonface_label; % 1 x N

    X = cat(2, X_pos, X_neg);  % K x 2N
    Y = cat(2, Y_pos, Y_neg);  % 1 x 2N
end