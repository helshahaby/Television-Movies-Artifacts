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

    convolution2dLayer(3 , 128 , 'Stride',1 , 'Padding' , 1 , 'Name' , 'Conv_1')
    batchNormalizationLayer('Name' , 'BN_1')
    reluLayer('Name' , 'Relu_1')

    convolution2dLayer(3 , 128 , 'Stride',1 , 'Padding' , 1 , 'Name' , 'Conv_2')
    batchNormalizationLayer('Name' , 'BN_2')
    reluLayer('Name' , 'Relu_2')
    
    maxPooling2dLayer(2 ,'Stride',2 , 'Name' , 'max_Pool_1')

    
    convolution2dLayer(3 , 256 , 'Stride',1 , 'Padding' , 1, 'Name' , 'Conv_3' )
    batchNormalizationLayer('Name' , 'BN_3')
    reluLayer('Name' , 'Relu_3')

    convolution2dLayer(3 , 256 , 'Stride',1 , 'Padding' , 1 , 'Name' , 'Conv_4')
    batchNormalizationLayer('Name' , 'BN_4')
    reluLayer('Name' , 'Relu_4')
    
    convolution2dLayer(3 , 256 , 'Stride',1 , 'Padding' , 1 , 'Name' , 'Conv_5')
    batchNormalizationLayer( 'Name' , 'BN_5')
    reluLayer('Name' , 'Relu_5')
    
    maxPooling2dLayer(2 ,'Stride',2, 'Name' , 'max_Pool_2') 
    
    
    
    convolution2dLayer(3 , 512 , 'Stride',1 , 'Padding' , 1 , 'Name' , 'Conv_6')
    batchNormalizationLayer('Name' , 'BN_6')
    reluLayer('Name' , 'Relu_6')

    convolution2dLayer(3 , 512 , 'Stride',1 , 'Padding' , 1 , 'Name' , 'Conv_7')
    batchNormalizationLayer('Name' , 'BN_7')
    reluLayer('Name' , 'Relu_7')
    
    convolution2dLayer(3 , 512 , 'Stride',1 , 'Padding' , 1 , 'Name' , 'Conv_8')
    batchNormalizationLayer('Name' , 'BN_8')
    reluLayer('Name' , 'Relu_8')
    
    maxPooling2dLayer(2 ,'Stride',2, 'Name' , 'max_Pool_3')     
    
    
    
    fullyConnectedLayer(1024, 'Name' , 'fc_1')
    
    fullyConnectedLayer(1024, 'Name' , 'fc_2')
        
    fullyConnectedLayer(2, 'Name' , 'fc_3')
    softmaxLayer('Name' , 'Soft_max')
    classificationLayer('Name' , 'Classify_layer')
];


options = trainingOptions('sgdm',...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.0001,...
    'MiniBatchSize',64,...
    'Verbose',false);


lgraph = layerGraph(layers);
figure
plot(lgraph);

net = trainNetwork(trainfilmData,layers,options);

predictedLabels = classify(net,valfilmData);
valLabels = valfilmData.Labels;

accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned 
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

VGG3_NET17b = net;
 
save VGG3_NET17b  % retrieve using load(net) 
VGG_NET accuracy = 0.xxxx 



[confusionMat,order] = confusionmat(valLabels, predictedLabels )

TP: true positive, TN: true negative, 
FP: false positive, FN: false negative 

precision = TP / (TP + FP) % for each class label 
precision = diag(confusionMat)./sum(confusionMat,2)
precisionMean = mean(precision)                                                                                                  % For average Precision

recall = sensitivity % for each class label 
recall =  diag(confusionMat)./sum(confusionMat,1)'
recallMean = mean(recall)  % For average Recall

F-score = 2 *TP /(2*TP + FP + FN) % for each class label 
F-score = 2 * Precision * Recall / (Precision + Recall)
f_Scores = 2 *( precision .* recall ) ./ ( precision + recall )

meanf_Scores = mean(f_Scores)

toc


VGG3_NET_ClassifyFramesToTextOrNot

labelCount =

  2×2 table

     Label      Count
    ________    _____

    Non_Text    22927
    Text        18581


accuracy =

    0.9995


confusionMat =

       12925           2
           9        8572


order = 

  2×1 categorical array

     Non_Text 
     Text 


precision =

    0.9998
    0.9990


precisionMean =

    0.9994


recall =

    0.9993
    0.9998


recallMean =

    0.9995


f_Scores =

    0.9996
    0.9994


meanf_Scores =

    0.9995

Elapsed time is 64752.246424 seconds.