% Goal
% Create Simple Deep Learning Network for Classification

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

%To reproduce the results in this example, set rng to '1'.
rng(1) % For reproducibility
[trainfilmData,valfilmData] = splitEachLabel(filmData,trainNumFiles,'randomize');

%     layer = convolution2dLayer(filterSize,numFilters,Name,Value)
%     layer = maxPooling2dLayer(poolSize,Name,Value)

layers = [
    imageInputLayer([40 200 1] , 'Name','input')

    convolution2dLayer(10 , 96 , 'Stride',2,'Padding',0, 'Name','Conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','Relu_1')


    maxPooling2dLayer(2,'Stride',2, 'Name','max_Pool_1')

    convolution2dLayer(3,256, 'Stride',1,'Padding',1, 'Name','Conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','Relu_2')

    maxPooling2dLayer(2,'Stride',2, 'Name','max_Pool_2')

    convolution2dLayer(3,384, 'Stride',1,'Padding',1, 'Name','Conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','Relu_3')

    convolution2dLayer(3,384, 'Stride',1,'Padding',1, 'Name','Conv_4')
    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','Relu_4')

    convolution2dLayer(3,256, 'Stride',1,'Padding',1, 'Name','Conv_5')
    batchNormalizationLayer('Name','BN_5')
    reluLayer('Name','Relu_5')
    
    maxPooling2dLayer(2,'Stride',2, 'Name','max_Pool_3')
    
    fullyConnectedLayer(1024, 'Name','fc_1')
    
    fullyConnectedLayer(2, 'Name','fc_2')
    softmaxLayer('Name','Softmax Layer')
    classificationLayer('Name','Classify Layer')
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

AlexNet17b = net;
 
save AlexNet17b  % retrieve using load(net)

% AlexTextNet15000_TrainMaxEpochs25LearnRate3e_2_layers = net;  
% save AlexTextNet15000_TrainMaxEpochs25LearnRate3e_2_layers  %accuracy = 0.6888 



% AlexTextNet15000_TrainMaxEpochs20LearnRate1e_4_layers = net;
% save AlexTextNet15000_TrainMaxEpochs20LearnRate1e_4_layers %accuracy = 0.9927

% testMem -> Accuracy = 0.9929 , 15000 , 0.00001

% AlexNetFix -> Accuracy , 13000 , 0.00001

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

% AlexNetFix
% ================
% labelCount = 
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
%     0.9813
% 
% 
% confusionMat =
% 
%         9880          47
%          243        5338
% 
% 
% order = 
% 
%      Non_Text 
%      Text 
% 
% 
% precision =
% 
%     0.9953
%     0.9565
% 
% 
% precisionMean =
% 
%     0.9759
% 
% 
% recall =
% 
%     0.9760
%     0.9913
% 
% 
% recallMean =
% 
%     0.9836
% 
% 
% f_Scores =
% 
%     0.9855
%     0.9736
% 
% 
% meanf_Scores =
% 
%     0.9795



% ALEX NET FIX 10 K
% 
% accuracy =
% 
%     0.9754
% 
% 
% confusionMat =
% 
%        12824         103
%          426        8155
% 
% 
% order = 
% 
%      Non_Text 
%      Text 
% 
% 
% precision =
% 
%     0.9920
%     0.9504
% 
% 
% precisionMean =
% 
%     0.9712
% 
% 
% recall =
% 
%     0.9678
%     0.9875
% 
% 
% recallMean =
% 
%     0.9777
% 
% 
% f_Scores =
% 
%     0.9798
%     0.9686
% 
% 
% meanf_Scores =
% 
%     0.9742



% AlexNet17b Batch Normalization

% accuracy =
% 
%     0.9987
% 
% 
% confusionMat =
% 
%        12908          19
%            9        8572
% 
% 
% order = 
% 
%   2×1 categorical array
% 
%      Non_Text 
%      Text 
%      
% precision =
% 
%     0.9985
%     0.9990
% 
% 
% precisionMean =
% 
%     0.9987
% 
% 
% recall =
% 
%     0.9993
%     0.9978
% 
% 
% recallMean =
% 
%     0.9985
% 
% 
% f_Scores =
% 
%     0.9989
%     0.9984
% 
% 
% meanf_Scores =
% 
%     0.9986
%  