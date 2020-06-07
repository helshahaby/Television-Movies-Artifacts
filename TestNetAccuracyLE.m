
clc
clear all
close all

rng('default')

% matlabroot = 'E:\MATLAB_R2016a_Installation\MATLAB\';
filmTestDatasetPath = 'E:\MATLAB_R2016a_Installation\MATLAB\FilmsTestDataSet';


% filmTestDatasetPath = fullfile(matlabroot,'FilmsDataSet','TestData');

filmTestData = imageDatastore(filmTestDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


valLabelsLe = filmTestData.Labels;

tic

load Le_NET17b

net = Le_NET17b;

% Le_NET17b.Layers

[YTestPred,scores] = classify(net,filmTestData);
toc

accuracy = sum(YTestPred == valLabelsLe )/numel(valLabelsLe)

[confusionMat,order] = confusionmat(valLabelsLe, YTestPred )

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


% Elapsed time is 22.931176 seconds.
% 
% accuracy =
% 
%     0.9688
% 
% 
% confusionMat =
% 
%         7774         278
%          154        5651
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
%     0.9655
%     0.9735
% 
% 
% precisionMean =
% 
%     0.9695
% 
% 
% recall =
% 
%     0.9806
%     0.9531
% 
% 
% recallMean =
% 
%     0.9668
% 
% 
% f_Scores =
% 
%     0.9730
%     0.9632
% 
% 
% meanf_Scores =
% 
%     0.9681
% 


