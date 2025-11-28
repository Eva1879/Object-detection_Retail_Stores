% Get the first 10 image file paths
imageFiles = testingData.imageFilename(1:276);  % Ensure testingData is a table

% Read the first 10 images into a cell array
images = cellfun(@imread, imageFiles, 'UniformOutput', false);

% Augment and detect each image individually without a loop
% Image 1
% img1 = augmentImage(images{1});
% [bboxes1, scores1, labels1] = detect(detector, img1);
% displayDetection(img1, bboxes1, labels1, 1);

% Augmentation: Add Gaussian noise
% noisyImage = imnoise(testImage, 'gaussian', 0, 0.05);  % 0.05 being most noise
% 
% % Augmentation: Adjust brightness/contrast
% augmentedImage = imadjust(noisyImage, stretchlim(noisyImage), []);

% Image 3
img100 = augmentImage(images{100});
[bboxes100, scores100, labels100] = detect(detector, img100);
displayDetection(img100, bboxes100, labels100, 100);

% ... repeat up to Image 10
% You can copy/paste and increment the index each time (img4, img5, etc.)

% Define augmentation as a separate function
function augImg = augmentImage(img)
    % Adjust brightness by applying a random factor between 1.1 and 1.5
    brightnessFactor = 1.1 + 0.4 * rand();  
    bright = imadjust(img, [], [], brightnessFactor);

    % Stretch contrast
    augImg = imadjust(bright, stretchlim(bright), []);


%     function augImg = augmentImage(img)
%     % Stretch contrast more aggressively by tightening input limits
%     lowHighIn = stretchlim(img, [0.01 0.99]);  % Keep only 1%-99% intensity range
%     augImg = imadjust(img, lowHighIn, []);     % Map that to full [0 1] range
% end

end



% Define display function
function displayDetection(img, bboxes, labels, index)
    figure;
    imshow(img);
    title(sprintf('Detection on Image %d (Augmented)', index));
    hold on;
    if isempty(bboxes)
        warning('No objects detected in image %d.', index);
    else
        for i = 1:size(bboxes, 1)
            rectangle('Position', bboxes(i, :), 'EdgeColor', 'r', 'LineWidth', 2);
            text(bboxes(i, 1), bboxes(i, 2) - 10, string(labels(i)), ...
                'Color', 'red', 'FontSize', 12, 'BackgroundColor', 'white', 'EdgeColor', 'black');
        end
    end
    hold off;
end
