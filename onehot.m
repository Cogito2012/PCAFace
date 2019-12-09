function [labels_onehot] = onehot(labels, cls_id)
% labels: 1 x N
% cls_id: 1xC, class identifiers (int)
% labels_onehot: C x N

labels_onehot = zeros(length(cls_id), length(labels));
for i=1:length(cls_id)
    inds = find(labels == cls_id(i));
    labels_onehot(i, inds) = 1;
end

end

