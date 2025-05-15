function [X_balanced, Y_balanced] = balanceDataset(X, Y, classes)
    % Balance dataset so all classes have equal number of samples
    
    counts = countcats(Y);
    minCount = min(counts);
    X_balanced = {};
    Y_balanced = categorical();

    for i = 1:length(classes)
        class = classes(i);
        idx = find(Y == class);
        idx = idx(randperm(length(idx), minCount)); % random minCount indices

        X_balanced = [X_balanced, X(idx)];
        Y_balanced = [Y_balanced; Y(idx)];
    end
end
