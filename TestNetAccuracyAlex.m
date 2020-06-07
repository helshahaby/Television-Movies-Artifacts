
clc
clear all
close all
rng('default')

% matlabroot = 'E:\MATLAB_R2016a_Installation\MATLAB\';
filmTestDatasetPath = 'E:\MATLAB_R2016a_Installation\MATLAB\FilmsTestDataSet';


% filmTestDatasetPath = fullfile(matlabroot,'FilmsDataSet','TestData');

filmTestData = imageDatastore(filmTestDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


valLabelsAlex = filmTestData.Labels;

tic
load AlexNet17b

net = AlexNet17b;

[YTestPred,scores] = classify(net,filmTestData);
toc

AlexNet17b.Layers

accuracy = sum(YTestPred == valLabelsAlex )/numel(valLabelsAlex)

[confusionMat,order] = confusionmat(valLabelsAlex, YTestPred )

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

% Elapsed time is 87.479269 seconds.
% 
% ans = 
% 
%   23x1 Layer array with layers:
% 
%      1   'imageinput'    Image Input             40x200x1 images with 'zerocenter' normalization
%      2   'conv_1'        Convolution             96 10x10x1 convolutions with stride [2  2] and padding [0  0  0  0]
%      3   'batchnorm_1'   Batch Normalization     Batch normalization with 96 channels
%      4   'relu_1'        ReLU                    ReLU
%      5   'maxpool_1'     Max Pooling             2x2 max pooling with stride [2  2] and padding [0  0  0  0]
%      6   'conv_2'        Convolution             256 3x3x96 convolutions with stride [1  1] and padding [1  1  1  1]
%      7   'batchnorm_2'   Batch Normalization     Batch normalization with 256 channels
%      8   'relu_2'        ReLU                    ReLU
%      9   'maxpool_2'     Max Pooling             2x2 max pooling with stride [2  2] and padding [0  0  0  0]
%     10   'conv_3'        Convolution             384 3x3x256 convolutions with stride [1  1] and padding [1  1  1  1]
%     11   'batchnorm_3'   Batch Normalization     Batch normalization with 384 channels
%     12   'relu_3'        ReLU                    ReLU
%     13   'conv_4'        Convolution             384 3x3x384 convolutions with stride [1  1] and padding [1  1  1  1]
%     14   'batchnorm_4'   Batch Normalization     Batch normalization with 384 channels
%     15   'relu_4'        ReLU                    ReLU
%     16   'conv_5'        Convolution             256 3x3x384 convolutions with stride [1  1] and padding [1  1  1  1]
%     17   'batchnorm_5'   Batch Normalization     Batch normalization with 256 channels
%     18   'relu_5'        ReLU                    ReLU
%     19   'maxpool_3'     Max Pooling             2x2 max pooling with stride [2  2] and padding [0  0  0  0]
%     20   'fc_1'          Fully Connected         1024 fully connected layer
%     21   'fc_2'          Fully Connected         2 fully connected layer
%     22   'softmax'       Softmax                 softmax
%     23   'classoutput'   Classification Output   crossentropyex with classes 'Non_Text' and 'Text'

% accuracy =
% 
%     0.9685
% 
% 
% confusionMat =
% 
%         7885         167
%          269        5536
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
%     0.9793
%     0.9537
% 
% 
% precisionMean =
% 
%     0.9665
% 
% 
% recall =
% 
%     0.9670
%     0.9707
% 
% 
% recallMean =
% 
%     0.9689
% 
% 
% f_Scores =
% 
%     0.9731
%     0.9621
% 
% 
% meanf_Scores =
% 
%     0.9676
