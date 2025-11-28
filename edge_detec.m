% Initialize edge density storage
edgeDensities = [];

for i = 1:size(bboxes, 1)
    box = round(bboxes(i, :));

    % Clamp box within image boundaries
    x1 = max(box(1), 1);
    y1 = max(box(2), 1);
    x2 = min(x1 + box(3) - 1, size(edges, 2));
    y2 = min(y1 + box(4) - 1, size(edges, 1));

    cropEdges = edges(y1:y2, x1:x2);

    if isempty(cropEdges)
        continue;
    end

    edgeDensity = sum(cropEdges(:)) / numel(cropEdges);
    edgeDensities = [edgeDensities; edgeDensity];
end

% Analyze Edge Density Distribution
figure;
histogram(edgeDensities, 20);
title('Edge Density Distribution');
xlabel('Edge Density');
ylabel('Frequency');

% Compute statistics
meanDensity = mean(edgeDensities);
medianDensity = median(edgeDensities);
disp(['Mean Edge Density: ', num2str(meanDensity)]);
disp(['Median Edge Density: ', num2str(medianDensity)]);

% Suggested threshold
suggestedThreshold = medianDensity;
disp(['Suggested Threshold: ', num2str(suggestedThreshold)]);
