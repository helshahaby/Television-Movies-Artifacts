
clc
clear all
close all

rng('default')

% matlabroot = 'E:\MATLAB_R2016a_Installation\MATLAB\';
filmTestDatasetPath = 'E:\MATLAB_R2016a_Installation\MATLAB\FilmsTestDataSet';


% filmTestDatasetPath = fullfile(matlabroot,'FilmsDataSet','TestData');

filmTestData = imageDatastore(filmTestDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


valLabelsZF = filmTestData.Labels;

tic

load ZF_NET17b

net = ZF_NET17b;

% ZF_NET17b.Layers

[YTestPred,scores] = classify(net,filmTestData);
toc

accuracy = sum(YTestPred == valLabelsZF )/numel(valLabelsZF)

[confusionMat,order] = confusionmat(valLabelsZF, YTestPred )

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

% 
% Elapsed time is 51.906470 seconds.
% 
% accuracy =
% 
%     0.9794
% 
% 
% confusionMat =
% 
%         7988          64
%          222        5583
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
%     0.9921
%     0.9618
% 
% 
% precisionMean =
% 
%     0.9769
% 
% 
% recall =
% 
%     0.9730
%     0.9887
% 
% 
% recallMean =
% 
%     0.9808
% 
% 
% f_Scores =
% 
%     0.9824
%     0.9750
% 
% 
% meanf_Scores =
% 
%     0.9787
% 
