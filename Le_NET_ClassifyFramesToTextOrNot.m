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
    imageInputLayer([40 200 1], 'Name' , 'input')

    convolution2dLayer(5 , 6 , 'Stride', 1, 'Name' , 'Conv_1')
    batchNormalizationLayer('Name' , 'BN_1')
    reluLayer( 'Name' , 'Relu_1')

    maxPooling2dLayer(2,'Stride',2 , 'Name' , 'maxPool_1')

    convolution2dLayer( 5 ,16, 'Stride', 1 , 'Name' , 'Conv_2')
    batchNormalizationLayer('Name' , 'BN_2')
    reluLayer('Name' , 'Relu_2')

    maxPooling2dLayer(2,'Stride',2, 'Name' , 'maxPool_2')

    
    fullyConnectedLayer( 120, 'Name' , 'fc_1')
    
    fullyConnectedLayer( 84, 'Name' , 'fc_2')
    
    fullyConnectedLayer(2, 'Name' , 'fc_3')
    softmaxLayer('Name' , 'Soft_Max')
    classificationLayer('Name' , 'Classify_Layer')
];


options = trainingOptions('sgdm',...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.0001,...
    'Verbose',false);

lgraph = layerGraph(layers);
figure
plot(lgraph);

net = trainNetwork(trainfilmData,layers,options);



predictedLabels = classify(net,valfilmData);
valLabels = valfilmData.Labels;

% accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned 
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

Le_NET17b_graph = net;
 
save Le_NET17b_graph  % retrieve using load(net) 
%accuracy = 0. -> Le_NET17b


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
% Le_NET_ClassifyFramesToTextOrNot
% 
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
%     0.9891
% 
% 
% confusionMat =
% 
%        12787         140
%           95        8486
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
%     0.9892
%     0.9889
% 
% 
% precisionMean =
% 
%     0.9890
% 
% 
% recall =
% 
%     0.9926
%     0.9838
% 
% 
% recallMean =
% 
%     0.9882
% 
% 
% f_Scores =
% 
%     0.9909
%     0.9863
% 
% 
% meanf_Scores =
% 
%     0.9886

