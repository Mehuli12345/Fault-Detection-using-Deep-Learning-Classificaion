function [YPred, accuracy, metrics] = evaluateModel(net, XTest, YTest, classes)
    % Evaluate network on test set and calculate metrics
    
    YPred = classify(net, XTest, 'MiniBatchSize', 16);
    accuracy = sum(YPred == YTest) / numel(YTest);

    cm = confusionmat(YTest, YPred);
    precision = diag(cm) ./ sum(cm, 2);
    recall = diag(cm) ./ sum(cm, 1)';
    f1 = 2 * (precision .* recall) ./ (precision + recall);
    mcc = (diag(cm) .* sum(sum(cm)) - sum(cm,2) .* sum(cm,1)') ./ ...
          sqrt(sum(cm,2) .* sum(cm,1)' .* (sum(sum(cm)) - sum(cm,2)) .* (sum(sum(cm)) - sum(cm,1)'));
    % Handle NaN in MCC (if denominator is zero)
    mcc(isnan(mcc)) = 0;

    macroP = mean(precision);
    macroR = mean(recall);
    macroF1 = mean(f1);
    macroM = mean(mcc);

    metrics = struct('precision', precision, 'recall', recall, 'f1', f1, 'mcc', mcc, ...
        'macroP', macroP, 'macroR', macroR, 'macroF1', macroF1, 'macroM', macroM);
end
