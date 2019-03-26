%% Load Dataset
dsTrain = imageDatastore('train', 'IncludeSubfolders', true, ...
                                  'LabelSource', 'foldernames');
dsValidate = imageDatastore('validate', 'IncludeSubfolders', true, ...
                                     'LabelSource', 'foldernames');
%% Resize Dataset
augTrain = augmentedImageDatastore([224 224], dsTrain);
augValidate = augmentedImageDatastore([224 224], dsValidate);
%% Load VGG-19 network
net = vgg19();
analyzeNetwork(net)
%% Replace last layers
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(dsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor', 20, ...
                                   'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

%% Network options
options = trainingOptions('sgdm', ...
                        'MiniBatchSize',10, ...
                        'MaxEpochs',6, ...
                        'InitialLearnRate',1e-4, ...
                        'Shuffle','every-epoch', ...
                        'ValidationData',augValidate, ...
                        'ValidationFrequency',3, ...
                        'Verbose',false, ...
                        'Plots','training-progress');
%% Train Network
vgg19LISC = trainNetwork(augTrain, layers, options);

%% Load Existing Model

load('vgg19-LISC.mat');
%% Dont forget to resize sample
% sample = imread('sample');
% sample = imresize(sample, [224 224]);
