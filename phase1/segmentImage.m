function res = segmentImage_general(X)

innerThreshold = 0.3686;
outerThreshold = 0.0980;

X = rgb2hsv(X); % conver RGB colors to HSV
p1 = X(:, :, 2) > innerThreshold; % inner nucleus part
p2 = X(:, :, 2) > outerThreshold; % outer fluid part

% clean artifacts around nucleus
se = strel('disk', 8);
p1 = imopen(p1, se); % Open mask with '8r' disk

% clean artifacts around fluid
se = strel('disk', 28);
p2 = imopen(p2, se); % Open mask with '28r' disk

p1 = im2uint8(p1);
p2 = im2uint8(p2);

p2(p2 == 255) = 150; % distinguish fluid from nucleus
res = p1 + p2; % combine components

end

