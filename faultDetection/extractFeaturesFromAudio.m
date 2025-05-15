function [X_all, Y_all, classes] = extractFeaturesFromAudio(datasetPath)
    % Extract MFCC + delta + delta-delta features from all audio files
    % Returns cell arrays X_all (features), Y_all (labels), and class names
    
    classes = ["Healthy", "Flywheel", "Riderbelt", "Piston", "LOV", "LIV", "Bearing", "NRV"];
    X_all = {};
    Y_all = {};
    
    for i = 1:length(classes)
        class = classes(i);
        audioFiles = dir(fullfile(datasetPath, char(class), '*.wav'));
        for j = 1:length(audioFiles)
            filename = fullfile(audioFiles(j).folder, audioFiles(j).name);
            [audioIn, fs] = audioread(filename);

            % Extract MFCC features
            coeffs = mfcc(audioIn, fs);
            delta = diff(coeffs, 1, 1);
            deltaDelta = diff(delta, 1, 1);

            % Resize delta and deltaDelta to match MFCC rows
            delta = [delta; zeros(1, size(delta, 2))];
            deltaDelta = [deltaDelta; zeros(2, size(deltaDelta, 2))];

            features = [coeffs'; delta'; deltaDelta'];
            X_all{end+1} = features;
            Y_all{end+1} = char(class);
        end
    end
    Y_all = categorical(Y_all);
end
