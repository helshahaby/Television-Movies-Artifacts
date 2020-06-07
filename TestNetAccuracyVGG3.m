
clc
clear all
close all

rng('default')

% matlabroot = 'E:\MATLAB_R2016a_Installation\MATLAB\';
filmTestDatasetPath = 'E:\MATLAB_R2016a_Installation\MATLAB\FilmsTestDataSet';


% filmTestDatasetPath = fullfile(matlabroot,'FilmsDataSet','TestData');

filmTestData = imageDatastore(filmTestDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


valLabelsVGG = filmTestData.Labels;

tic

load VGG3_NET17b

net = VGG3_NET17b;

% VGG3_NET17b.Layers

[YTestPred,scores] = classify(net,filmTestData);
toc

accuracy = sum(YTestPred == valLabelsVGG )/numel(valLabelsVGG)

[confusionMat,order] = confusionmat(valLabelsVGG, YTestPred )

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


% Elapsed time is 282.777067 seconds.
% 
% accuracy =
% 
%     0.9804
% 
% 
% confusionMat =
% 
%         8023          29
%          243        5562
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
%     0.9964
%     0.9581
% 
% 
% precisionMean =
% 
%     0.9773
% 
% 
% recall =
% 
%     0.9706
%     0.9948
% 
% 
% recallMean =
% 
%     0.9827
% 
% 
% f_Scores =
% 
%     0.9833
%     0.9761
% 
% 
% meanf_Scores =
% 
%     0.9797

