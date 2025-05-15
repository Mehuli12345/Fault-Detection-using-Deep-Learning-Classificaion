function plotROC(YTrue, scores, classes)
    % Plot multi-class ROC curves

    figure;
    hold on;
    colors = lines(numel(classes));
    for i = 1:numel(classes)
        [X, Y, ~, AUC] = perfcurve(YTrue == classes(i), scores(:, i), true);
        plot(X, Y, 'Color', colors(i,:), 'LineWidth', 2, 'DisplayName', ...
            sprintf('%s (AUC = %.2f)', classes(i), AUC));
    end
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('Multi-Class ROC Curves');
    legend('Location', 'Best');
    grid on;
    hold off;
end
