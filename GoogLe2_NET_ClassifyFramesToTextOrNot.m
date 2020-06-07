% Goal
% Create Simple Deep Learning Network for Classification

tic

clc
clear all
close all

matlabroot = 'E:\MATLAB_R2016a_Installation\MATLAB\';
filmDatasetPath = 'E:\MATLAB_R2016a_Installation\MATLAB\FilmsDataSet';


filmDatasetPath = fullfile(matlabroot,'FilmsDataSet');
filmData = imageDatastore(filmDatasetPath,...
'IncludeSubfolders',true,'LabelSource','foldernames');

% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(filmData.Files{perm(i)});
% end

labelCount = countEachLabel(filmData)

% img = readimage(filmData,1);
% [m , n ,b ] = size(img)

trainNumFiles = 10000;
rng('default') % For reproducibility 
[trainfilmData,valfilmData] = splitEachLabel(filmData,trainNumFiles,'randomize');

%     layer = convolution2dLayer(filterSize,numFilters,Name,Value)
%     layer = maxPooling2dLayer(poolSize,Name,Value)

layers = [
    imageInputLayer([40 200 1]  ,'Name','input')

    convolution2dLayer(5 , 64 , 'Stride', 1 ,'Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')

    convolution2dLayer( 1 ,64, 'Stride', 1 ,'Name','conv_2' )
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')

    convolution2dLayer( 5 ,192, 'Stride',1 ,'Padding',0,'Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    maxPooling2dLayer(2,'Stride',2 ,'Name','maxPool')
        
    averagePooling2dLayer(2,'Stride',2 ,'Name','avgPool')
        
    fullyConnectedLayer( 1024 ,'Name','fc1')
    
    fullyConnectedLayer( 1024 ,'Name','fc2')
    
    fullyConnectedLayer(2 ,'Name','fc3')
    
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')
    
];


in1_3a = [
    convolution2dLayer(1 , 64 , 'Stride', 1 ,'Name','Inception_3a_in1_1x1');
    batchNormalizationLayer('Name','BN_Inception_3a_in1_1x1')
    reluLayer('Name','relu_Inception_3a_in1_1x1')
    ];

in2_3a = [
    convolution2dLayer(1 , 96 , 'Stride', 1 ,'Name','Inception_3a_in2_3x3_reduce')
    batchNormalizationLayer('Name','BN_Inception_3a_in2_3x3_reduce')
    reluLayer('Name','relu_Inception_3a_in2_3x3_reduce')
    convolution2dLayer(3 , 128 , 'Stride', 1 ,'Name','Inception_3a_in3_1x1')
    batchNormalizationLayer('Name','BN_Inception_3a_in3_1x1')
    reluLayer('Name','relu_Inception_3a_in3_1x1')
    ];

in3_3a = [
    convolution2dLayer(1 , 16 , 'Stride', 1 ,'Name','Inception_3a_in4_5x5_reduce')
    batchNormalizationLayer('Name','BN_Inception_3a_in4_5x5_reduce')
    reluLayer('Name','relu_Inception_3a_in4_5x5_reduce')    
    convolution2dLayer(5 , 32 , 'Stride', 1 ,'Name','Inception_3a_5x5')
    batchNormalizationLayer('Name','BN_Inception_3a_5x5')
    reluLayer('Name','relu_Inception_3a_5x5')    
    ];

in4_3a = [
    maxPooling2dLayer(2,'Stride',2 ,'Name','Inception_3a_maxPool')
    convolution2dLayer(1 , 32 , 'Stride', 1 ,'Name','Inception_3a_maxPool_proj')
    batchNormalizationLayer('Name','BN_Inception_3a_maxPool_proj')
    reluLayer('Name','relu_Inception_3a_maxPool_proj')
    ];


in1_3b = [
    convolution2dLayer(1 , 128 , 'Stride', 1 ,'Name','Inception_3b_in1_1x1');
    batchNormalizationLayer('Name','BN_Inception_3b_in1_1x1')
    reluLayer('Name','relu_Inception_3b_in1_1x1')
    ];

in2_3b = [
    convolution2dLayer(1 , 128 , 'Stride', 1 ,'Name','Inception_3b_in2_3x3_reduce')
    batchNormalizationLayer('Name','BN_Inception_3b_in2_3x3_reduce')
    reluLayer('Name','relu_Inception_3b_in2_3x3_reduce')
    convolution2dLayer(3 , 192 , 'Stride', 1 ,'Name','Inception_3b_in3_1x1')
    batchNormalizationLayer('Name','BN_Inception_3b_in3_1x1')
    reluLayer('Name','relu_Inception_3b_in3_1x1')
    ];

in3_3b = [
    convolution2dLayer(1 , 32 , 'Stride', 1 ,'Name','Inception_3b_in4_5x5_reduce')
    batchNormalizationLayer('Name','BN_Inception_3b_in4_5x5_reduce')
    reluLayer('Name','relu_Inception_3b_in4_5x5_reduce')    
    convolution2dLayer(5 , 96 , 'Stride', 1 ,'Name','Inception_3b_5x5')
    batchNormalizationLayer('Name','BN_Inception_3b_5x5')
    reluLayer('Name','relu_Inception_3b_5x5')
    ];

in4_3b = [
    maxPooling2dLayer(2,'Stride',2 ,'Name','Inception_3b_maxPool')
    convolution2dLayer(1 , 64 , 'Stride', 1 ,'Name','Inception_3b_maxPool_proj')
    batchNormalizationLayer('Name','BN_Inception_3b_maxPool_proj')
    reluLayer('Name','relu_Inception_3b_maxPool_proj')
    ];


lgraph = layerGraph(layers);
lgraph = addLayers(lgraph,in1_3a);
lgraph = addLayers(lgraph,in2_3a);
lgraph = addLayers(lgraph,in3_3a);
lgraph = addLayers(lgraph,in4_3a);

lgraph = addLayers(lgraph,in1_3b);
lgraph = addLayers(lgraph,in2_3b);
lgraph = addLayers(lgraph,in3_3b);
lgraph = addLayers(lgraph,in4_3b);

concat_1 =   depthConcatenationLayer(4,'Name','concat_1');

concat_2 =   depthConcatenationLayer(4,'Name','concat_2');
  
lgraph = addLayers(lgraph,concat_1);
lgraph = addLayers(lgraph,concat_2);

lgraph = disconnectLayers(lgraph,'maxPool','avgPool');


lgraph = connectLayers(lgraph,'maxPool','Inception_3a_in1_1x1');
lgraph = connectLayers(lgraph,'maxPool','Inception_3a_in2_3x3_reduce');
lgraph = connectLayers(lgraph,'maxPool','Inception_3a_in4_5x5_reduce');
lgraph = connectLayers(lgraph,'maxPool','Inception_3a_maxPool');



lgraph = connectLayers(lgraph,'relu_Inception_3a_in1_1x1','concat_1/in1');
lgraph = connectLayers(lgraph,'relu_Inception_3a_in3_1x1','concat_1/in2');
lgraph = connectLayers(lgraph,'relu_Inception_3a_5x5','concat_1/in3');
lgraph = connectLayers(lgraph,'relu_Inception_3a_maxPool_proj','concat_1/in4');



lgraph = connectLayers(lgraph,'concat_1','Inception_3b_in1_1x1');
lgraph = connectLayers(lgraph,'concat_1','Inception_3b_in2_3x3_reduce');
lgraph = connectLayers(lgraph,'concat_1','Inception_3b_in4_5x5_reduce');
lgraph = connectLayers(lgraph,'concat_1','Inception_3b_maxPool');


lgraph = connectLayers(lgraph,'relu_Inception_3b_in1_1x1','concat_2/in1');
lgraph = connectLayers(lgraph,'relu_Inception_3b_in3_1x1','concat_2/in2');
lgraph = connectLayers(lgraph,'relu_Inception_3b_5x5','concat_2/in3');
lgraph = connectLayers(lgraph,'relu_Inception_3b_maxPool_proj','concat_2/in4');

lgraph = connectLayers(lgraph,'concat_2','avgPool');

figure
plot(lgraph);

options = trainingOptions('sgdm',...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.0001,...
    'MiniBatchSize', 64, ...
    'Verbose',false);

net = trainNetwork(trainfilmData,layers,options);

predictedLabels = classify(net,valfilmData);
valLabels = valfilmData.Labels;

% accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned 
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

GoogLe2_NET17b = net;
 
save GoogLe2_NET17b  % retrieve using load(net) 

[confusionMat,order] = confusionmat(valLabels, predictedLabels )

% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative 

% precision = TP / (TP + FP) % for each class label 
precision = diag(confusionMat)./sum(confusionMat,2)
precisionMean = mean(precision)  % For average Precision

% recall = sensitivity % for each class label 
recall =  diag(confusionMat)./sum(confusionMat,1)'
recallMean = mean(recall)  % For average Recall

% F-score = 2 *TP /(2*TP + FP + FN) % for each class label 
% F-score = 2 * Precision * Recall / (Precision + Recall)
f_Scores = 2 *( precision .* recall ) ./ ( precision + recall )

meanf_Scores = mean(f_Scores)
toc

% labelCount =
% 
%   2×2 table
% 
%      Label      Count
%     ________    _____
% 
%     Non_Text    22927
%     Text        18581
% 
% 
% accuracy =
% 
%     0.9983
% 
% 
% confusionMat =
% 
%        12911          16
%           21        8560
% 
% 
% order = 
% 
%   2×1 categorical array
% 
%      Non_Text 
%      Text 
% 
% 
% precision =
% 
%     0.9988
%     0.9976
% 
% 
% precisionMean =
% 
%     0.9982
% 
% 
% recall =
% 
%     0.9984
%     0.9981
% 
% 
% recallMean =
% 
%     0.9983
% 
% 
% f_Scores =
% 
%     0.9986
%     0.9978
% 
% 
% meanf_Scores =
% 
%     0.9982
% 
% Elapsed time is 10178.298019 seconds.