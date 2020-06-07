
clc
clear all
close all

rng('default')

% matlabroot = 'E:\MATLAB_R2016a_Installation\MATLAB\';
filmTestDatasetPath = 'E:\MATLAB_R2016a_Installation\MATLAB\FilmsTestDataSet';


% filmTestDatasetPath = fullfile(matlabroot,'FilmsDataSet','TestData');

filmTestData = imageDatastore(filmTestDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');


valLabelsGoogLe = filmTestData.Labels;

tic

load GoogLe_NET17b

net = GoogLe_NET17b;

GoogLe_NET17b.Layers

[YTestPred,scores] = classify(net,filmTestData);
toc

accuracy = sum(YTestPred == valLabelsGoogLe )/numel(valLabelsGoogLe)

[confusionMat,order] = confusionmat(valLabelsGoogLe, YTestPred )

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
% Elapsed time is 127.592966 seconds.
% 
% accuracy =
% 
%     0.9695
% 
% 
% confusionMat =
% 
%         7764         288
%          135        5670
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
%     0.9642
%     0.9767
% 
% 
% precisionMean =
% 
%     0.9705
% 
% 
% recall =
% 
%     0.9829
%     0.9517
% 
% 
% recallMean =
% 
%     0.9673
% 
% 
% f_Scores =
% 
%     0.9735
%     0.9640
% 
% 
% meanf_Scores =
% 
%     0.9688
% 
