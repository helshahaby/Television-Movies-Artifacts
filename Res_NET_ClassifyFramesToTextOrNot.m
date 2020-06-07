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

    convolution2dLayer(5 , 8 , 'Stride', 1 ,'Name','conv_1')
    batchNormalizationLayer('Name','BN_1')

    maxPooling2dLayer(2,'Stride',2 ,'Name','maxPool')
       
    convolution2dLayer(1 , 8 , 'Stride', 1 ,'Name','conv_2')
    batchNormalizationLayer('Name','BN_2')    
    
    convolution2dLayer(3 , 8 , 'Stride', 1 ,'Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    
    convolution2dLayer(1 , 32 , 'Stride', 1 ,'Name','conv_4')
    batchNormalizationLayer('Name','BN_4')
    
    
    reluLayer('Name','relu_1')
    
    
    convolution2dLayer(1 , 8 , 'Stride', 1 ,'Name','conv_5')
    batchNormalizationLayer('Name','BN_5')
    
    convolution2dLayer(3 , 8 , 'Stride', 1 ,'Name','conv_6')
    batchNormalizationLayer('Name','BN_6')
    
    convolution2dLayer(1 , 32 , 'Stride', 1 ,'Name','conv_7')
    batchNormalizationLayer('Name','BN_7')
    
    
    reluLayer('Name','relu_2')
    
    
    convolution2dLayer(1 , 16 , 'Stride', 1 ,'Name','conv_8')
    batchNormalizationLayer('Name','BN_8')
    
    convolution2dLayer(3 , 16 , 'Stride', 1 ,'Name','conv_9')
    batchNormalizationLayer('Name','BN_9') 
    
    convolution2dLayer(1 , 64 , 'Stride', 1 ,'Name','conv_10')
    batchNormalizationLayer('Name','BN_10')
    
    
    reluLayer('Name','relu_3')
    
    
    convolution2dLayer(1 , 16 , 'Stride', 1 ,'Name','conv_11')
    batchNormalizationLayer('Name','BN_11')
    
    convolution2dLayer(3 , 16 , 'Stride', 1 ,'Name','conv_12')
    batchNormalizationLayer('Name','BN_12')
    
    convolution2dLayer(1 , 64 , 'Stride', 1 ,'Name','conv_13')
    batchNormalizationLayer('Name','BN_13')
    
    
    reluLayer('Name','relu_4')
    
    
    convolution2dLayer(1 , 16 , 'Stride', 1 ,'Name','conv_14')
    batchNormalizationLayer('Name','BN_14')
    
    convolution2dLayer(3 , 16 , 'Stride', 1 ,'Name','conv_15')
    batchNormalizationLayer('Name','BN_15')
    
    convolution2dLayer(1 , 64 , 'Stride', 1 ,'Name','conv_16')
    batchNormalizationLayer('Name','BN_16')
    
        
    fullyConnectedLayer( 1024 ,'Name','fc1')
    
    fullyConnectedLayer( 1024 ,'Name','fc2')
    
    fullyConnectedLayer(2 ,'Name','fc3')
    
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')
    
];


in2_1x1_a = [
    convolution2dLayer(1 , 32 , 'Stride', 1 ,'Name','Inception_a_in2_1x1')
    batchNormalizationLayer('Name','BN_Incep1_a')
    ];

in2_1x1_b = [
    convolution2dLayer(1 , 64 , 'Stride', 1 ,'Name','Inception_b_in2_1x1')
    batchNormalizationLayer('Name','BN_Incep1_b')
    ];


lgraph = layerGraph(layers);
lgraph = addLayers(lgraph,in2_1x1_a);
lgraph = addLayers(lgraph,in2_1x1_b);


concat_1 =   depthConcatenationLayer(2,'Name','concat_1');

concat_2 =   depthConcatenationLayer(2,'Name','concat_2');
  
lgraph = addLayers(lgraph,concat_1);
lgraph = addLayers(lgraph,concat_2);

lgraph = disconnectLayers(lgraph,'maxPool','BN_2');
lgraph = disconnectLayers(lgraph,'relu_2','BN_7');
lgraph = disconnectLayers(lgraph,'relu_3','BN_10');


lgraph = disconnectLayers(lgraph,'BN_4','relu_1');
lgraph = connectLayers(lgraph,'concat_1','relu_1');


lgraph = connectLayers(lgraph,'maxPool','Inception_a_in2_1x1');

lgraph = connectLayers(lgraph,'BN_4','concat_1/in1');
lgraph = connectLayers(lgraph,'BN_Incep1_a','concat_1/in2');


lgraph = connectLayers(lgraph,'BN_Incep1_b','concat_2/in1');
lgraph = connectLayers(lgraph,'BN_13','concat_2/in2');

lgraph = disconnectLayers(lgraph,'relu_4','concat_2/in1');
lgraph = disconnectLayers(lgraph,'relu_4','concat_2/in2');

add_1 = additionLayer(2,'Name','add_1');
add_2 = additionLayer(2,'Name','add_2');
add_3 = additionLayer(2,'Name','add_3');

lgraph = addLayers(lgraph,add_1);
lgraph = addLayers(lgraph,add_2);
lgraph = addLayers(lgraph,add_3);

lgraph = disconnectLayers(lgraph,'BN_7','relu_2');
lgraph = connectLayers(lgraph,'add_1','relu_2');

lgraph = disconnectLayers(lgraph,'BN_10','relu_3');
lgraph = connectLayers(lgraph,'add_2','relu_3');

lgraph = disconnectLayers(lgraph,'BN_13','relu_4');
lgraph = connectLayers(lgraph,'add_3','relu_4');

lgraph = connectLayers(lgraph,'relu_1','add_1/in1');
lgraph = connectLayers(lgraph,'BN_7','add_1/in2');

lgraph = connectLayers(lgraph,'relu_3', 'Inception_b_in2_1x1');

lgraph = connectLayers(lgraph,'relu_2','add_2/in1');
lgraph = connectLayers(lgraph,'BN_10','add_2/in2');

lgraph = connectLayers(lgraph,'relu_3','add_3/in1');
lgraph = connectLayers(lgraph,'concat_2','add_3/in2');

        
figure
plot(lgraph);

options = trainingOptions('sgdm',...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.00002,...  
    'Verbose',false);

net = trainNetwork(trainfilmData,layers,options);

predictedLabels = classify(net,valfilmData);
valLabels = valfilmData.Labels;

% accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned 
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

Res_NET17b = net;
 
save Res_NET17b  % retrieve using load(net) 

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
%     0.9989
% 
% 
% confusionMat =
% 
%        12923           4
%           20        8561
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
%     0.9997
%     0.9977
% 
% 
% precisionMean =
% 
%     0.9987
% 
% 
% recall =
% 
%     0.9985
%     0.9995
% 
% 
% recallMean =
% 
%     0.9990
% 
% 
% f_Scores =
% 
%     0.9991
%     0.9986
% 
% 
% meanf_Scores =
% 
%     0.9988
% 
% Elapsed time is 1926.976346 seconds.
