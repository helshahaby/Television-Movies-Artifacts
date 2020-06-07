
clc
clear all
close all

rng('default')

% matlabroot = 'E:\MATLAB_R2016a_Installation\MATLAB\';
filmTestDatasetPath = 'E:\MATLAB_R2016a_Installation\MATLAB\FilmsTestDataSet';


% filmTestDatasetPath = fullfile(matlabroot,'FilmsDataSet','TestData');

filmTestData = imageDatastore(filmTestDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


valLabelsRes = filmTestData.Labels;

tic

load Res_NET17b

net = Res_NET17b;

% Res_NET17b.Layers
[YTestPred,scores] = classify(net,filmTestData);
toc

accuracy = sum(YTestPred == valLabelsRes )/numel(valLabelsRes)

[confusionMat,order] = confusionmat(valLabelsRes, YTestPred )

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

% Elapsed time is 124.299936 seconds.
% 
% accuracy =
% 
%     0.9758
% 
% 
% confusionMat =
% 
%         7982          70
%          266        5539
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
%     0.9913
%     0.9542
% 
% 
% precisionMean =
% 
%     0.9727
% 
% 
% recall =
% 
%     0.9677
%     0.9875
% 
% 
% recallMean =
% 
%     0.9776
% 
% 
% f_Scores =
% 
%     0.9794
%     0.9706
% 
% 
% meanf_Scores =
% 
%     0.9750
% 

