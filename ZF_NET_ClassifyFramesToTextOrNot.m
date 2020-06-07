% Goal
% Create Simple Deep Learning Network for Classification

tic

clc
clear all
close all

matlabroot = 'E:\MATLAB_R2016a_Installation\MATLAB\';
filmDatasetPath = 'E:\MATLAB_R2016a_Installation\MATLAB\FilmsDataSet_ManyLang';


filmDatasetPath = fullfile(matlabroot,'FilmsDataSet');
filmData = imageDatastore(filmDatasetPath,...
'IncludeSubfolders',true,'LabelSource','folderNames');

% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(filmData.Files{perm(i)});
% end

labelCount = countEachLabel(filmData)

% img = readimage(filmData,1);
% [m , n ,b ] = size(img)

trainNumFiles = 15000;
rng('default') % For reproducibility 
[trainfilmData,valfilmData] = splitEachLabel(filmData,trainNumFiles,'randomize');

%     layer = convolution2dLayer(filterSize,numFilters,'Name',Value)
%     layer = maxPooling2dLayer(poolSize,'Name',Value)

layers = [
    imageInputLayer([40 200 1],'Name' , 'input')

    convolution2dLayer(3 , 64 , 'Stride', 1 , 'Name' , 'Conv_1')
    batchNormalizationLayer('Name' , 'BN_1')
    reluLayer( 'Name' , 'Relu_1')

    maxPooling2dLayer(2,'Stride',2 , 'Name' , 'maxPool_1')

    convolution2dLayer( 2 ,128, 'Stride', 1 , 'Name' , 'Conv_2')
    batchNormalizationLayer('Name' , 'BN_2')
    reluLayer('Name' , 'Relu_2')

    maxPooling2dLayer(2,'Stride',2 , 'Name' , 'maxPool_2')

    convolution2dLayer( 2 ,256, 'Stride',1 ,'Padding',0, 'Name' , 'Conv_3')
    batchNormalizationLayer('Name' , 'BN_3')
    reluLayer('Name' , 'Relu_3')

    convolution2dLayer( 2 ,512,'Stride',1 ,'Padding',0, 'Name' , 'Conv_4')
    batchNormalizationLayer( 'Name' , 'BN_4')
    reluLayer('Name' , 'Relu_4')

    convolution2dLayer( 2 ,256, 'Stride',1 ,'Padding',0, 'Name' , 'Conv_5')
    batchNormalizationLayer('Name' , 'BN_5')
    reluLayer('Name' , 'Relu_5')
    
    maxPooling2dLayer( 3 ,'Stride',1, 'Name' , 'maxPool_3')
    
    fullyConnectedLayer( 1024, 'Name' , 'fc_1')
    
    fullyConnectedLayer( 1024, 'Name' , 'fc_2')
    
    fullyConnectedLayer(2, 'Name' , 'fc_3')
    softmaxLayer('Name' , 'SoftMax')
    classificationLayer('Name' , 'ClassLayer')
];


lgraph = layerGraph(layers);
figure
plot(lgraph);

options = trainingOptions('sgdm',...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.0001,...
    'Verbose',false);

net = trainNetwork(trainfilmData,layers,options);



predictedLabels = classify(net,valfilmData);
valLabels = valfilmData.Labels;

% accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned 
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

ZF_NET17b = net;
 
save ZF_NET17b  % retrieve using load(net) 
%accuracy = 0.9839 -> ZF_NET
%accuracy = 0. -> ZF_NET17b


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

