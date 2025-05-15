clc; clear; close all;

datasetPath = 'C:\Users\lenovo\AppData\Local\Temp\aircompressordataset\AirCompressorDataset';

fprintf('Extracting features...\n');
[X_all, Y_all, classes] = extractFeaturesFromAudio(datasetPath);

fprintf('Balancing dataset...\n');
[X_all, Y_all] = balanceDataset(X_all, Y_all, classes);

fprintf('Splitting data (5-fold CV)...\n');
cv = cvpartition(Y_all, 'KFold', 5);
XTrain = X_all(training(cv,1));
YTrain = Y_all(training(cv,1));
XTest  = X_all(test(cv,1));
YTest  = Y_all(test(cv,1));

fprintf('Building LSTM network...\n');
inputSize = size(X_all{1}, 1);
numClasses = numel(classes);
layers = buildLSTM(inputSize, numClasses);

options = trainingOptions('adam', ...
    'MaxEpochs', 60, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 0.005, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 15, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

fprintf('Training network...\n');
tic;
net = trainNetwork(XTrain, YTrain, layers, options);
toc;

fprintf('Evaluating network...\n');
[YPred, accuracy, metrics] = evaluateModel(net, XTest, YTest, classes);
fprintf('Test Accuracy: %.2f%%\n', accuracy*100);

disp('Class-wise metrics:');
for i = 1:numel(classes)
    fprintf('%s -> Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%, MCC: %.2f\n', ...
        classes(i), metrics.precision(i)*100, metrics.recall(i)*100, ...
        metrics.f1(i)*100, metrics.mcc(i));
end

figure; confusionchart(YTest, YPred); title('Confusion Matrix');

% Predict class probabilities for ROC
scoreCell = predict(net, XTest, 'MiniBatchSize', 16);
numSamples = numel(scoreCell);
scores = zeros(numSamples, numClasses);
for i = 1:numSamples
    scores(i, :) = scoreCell{i}(:)';
end

plotROC(YTest, scores, classes);
