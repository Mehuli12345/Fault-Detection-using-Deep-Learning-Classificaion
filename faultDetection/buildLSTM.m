function layers = buildLSTM(inputSize, numClasses)
    % Build LSTM network architecture for classification
    
    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(100, 'OutputMode', 'last')
        batchNormalizationLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
end
