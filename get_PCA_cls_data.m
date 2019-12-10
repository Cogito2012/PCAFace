function [X, Y, P] = get_PCA_cls_data(face_data, face_label, nonface_data, nonface_label, type, feat_dim, shared, expand, P)

if shared
    data = cat(2, face_data, nonface_data);  % (D1 x 2N)
    if isempty(P)
        [P, ~, ~] = facePCA(data, mean(data, 2), type);
        P = P(:, 1:feat_dim); % D1 x K (K<<D1)
    end
    X = P' * data;  % K x 2N
    if expand
        X = poly_expand(X);
    end
    Y = cat(2, face_label, nonface_label);   % 1 x 2N
else
    if isempty(P)
        [P1, ~, ~] = facePCA(face_data, mean(face_data, 2), type); 
        P1 = P1(:, 1:feat_dim); % D1 x K (K<<D1)
        [P2, ~, ~] = facePCA(nonface_data, mean(nonface_data, 2), type);
        P2 = P2(:, 1:feat_dim); % D2 x K (K<<D2)
    else
        P1 = P(1:size(face_data,1), :);
        P2 = P(size(face_data,1)+1:end, :);
    end
    
    % positive
    X_pos = P1' * face_data;  % K x N
    if expand
        X_pos = poly_expand(X_pos);
    end
    Y_pos = face_label; % 1 x N

    % negative
    X_neg = P2' * nonface_data;  % K x N
    if expand
        X_neg = poly_expand(X_neg);
    end
    Y_neg = nonface_label; % 1 x N

    X = cat(2, X_pos, X_neg);  % K x 2N
    Y = cat(2, Y_pos, Y_neg);  % 1 x 2N
    P = cat(1, P1, P2);
end
