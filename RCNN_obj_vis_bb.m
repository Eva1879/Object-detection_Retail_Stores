%% Step 1: Define Paths and Initialize Data
% Define paths to train, test, and valid folders
baseFolder = 'C:\Users\grace\Downloads\dataset';
folders = {'train', 'test', 'valid'};

% Initialize a structure to store data tables for each set
datasetTables = struct();

% Iterate over each dataset folder (train, test, valid)
for f = 1:numel(folders)
    datasetFolder = fullfile(baseFolder, folders{f});
    
    % Step 1: Load class names from _darknet.labels (only need to load once)
    if f == 1
        labelFilePath = fullfile(datasetFolder, '_darknet.labels');
        classNames = readlines(labelFilePath); % Read class names into a string array
        disp('Class names in _darknet.labels:');
        disp(classNames);
    end

    % Step 2: Create an ImageDatastore for images
    imageFiles = dir(fullfile(datasetFolder, '*.jpg')); % Get all .jpg files
    imagePaths = fullfile(datasetFolder, {imageFiles.name}); % Full paths to images
    imds = imageDatastore(imagePaths); % Create ImageDatastore

    % Step 3: Load annotations from .txt files
    annotationFiles = dir(fullfile(datasetFolder, '*.txt')); % Get all .txt files
    annotationFiles = annotationFiles(~strcmp({annotationFiles.name}, '_darknet.labels')); % Exclude _darknet.labels
    annotationPaths = fullfile(datasetFolder, {annotationFiles.name}); % Full paths to annotations

    % Initialize a cell array to store annotations
    annotations = cell(numel(annotationPaths), 1);

    % Read annotations from each .txt file
    for i = 1:numel(annotationPaths)
        fileID = fopen(annotationPaths{i}, 'r');
        annotations{i} = fscanf(fileID, '%c'); % Read the entire file as text
        fclose(fileID);
    end

    % Step 4: Verify alignment between images and annotations
    [~, imageNames] = cellfun(@fileparts, imagePaths, 'UniformOutput', false);
    [~, annotationNames] = cellfun(@fileparts, annotationPaths, 'UniformOutput', false);

    if ~isequal(imageNames, annotationNames)
        error('Image and annotation files in %s do not match. Check file names.', folders{f});
    end

    % Step 5: Prepare data table for this dataset
    firstImage = imread(imagePaths{1});
    imageSize = size(firstImage); % imageSize = [height, width, channels]
    imageSize = imageSize(1:2); % Extract only height and width

    % Initialize a table to store image file names, bounding boxes, and labels
    data = table('Size', [numel(imagePaths), 3], ...
                 'VariableTypes', {'string', 'cell', 'cell'}, ... % Ensure 'boxes' and 'labels' are cell arrays
                 'VariableNames', {'imageFilename', 'boxes', 'labels'});

    % Parse annotations and populate the table
    for i = 1:numel(imagePaths)
        annotation = annotations{i};
        annotationLines = splitlines(annotation); % Split into individual lines

        % Initialize arrays to store bounding boxes and labels for this image
        boxes = [];
        labels = [];

        % Parse each line of the annotation
        for j = 1:numel(annotationLines)
            if isempty(annotationLines{j})
                continue; % Skip empty lines
            end

            % Split the line into components (YOLO format: class_id x_center y_center width height)
            components = str2double(strsplit(annotationLines{j}));
            class_id = components(1) + 1; % YOLO class IDs start from 0, MATLAB starts from 1
            x_center = components(2);
            y_center = components(3);
            width = components(4);
            height = components(5);

            % Convert YOLO format to [x_min y_min width height]
            x_min = (x_center - width / 2) * imageSize(2); % imageSize(2) = width
            y_min = (y_center - height / 2) * imageSize(1); % imageSize(1) = height
            boxes = [boxes; x_min y_min width * imageSize(2) height * imageSize(1)];

            % Add the corresponding label
            labels = [labels; classNames(class_id)];
        end
        
        % Debug: Display labels for the current image
        %disp(['Labels for image ', num2str(i), ':']);
        %disp(labels);

        % Add the data to the table
        data.imageFilename(i) = imagePaths{i};
        data.boxes{i} = {boxes}; % Wrap in a cell array
        data.labels{i} = {categorical(labels, classNames)}; % Wrap in a cell array
    end

    % Store the data table in the datasetTables structure
    datasetTables.(folders{f}) = data;

    % Display a preview of the first few rows
    fprintf('Dataset: %s\n', folders{f});
    disp(head(data));
end

% Load the training data
trainingData = datasetTables.train;
testingData = datasetTables.test;
validData = datasetTables.valid;

%% Step 2: Verify Bounding Boxes and Labels
% Verify bounding box format
numRows = size(trainingData, 1); % Use size instead of height
disp("Number of rows in trainData: " + numRows);
for i = 1:numRows
    if isempty(trainingData.boxes{i})
        error('Bounding boxes for image %s are missing.', trainingData.imageFilename(i));
    end
    bbox = trainingData.boxes{i}{1}; % Extract bounding boxes for the current image
    if size(bbox, 2) ~= 4
        error('Bounding boxes for image %s are not in [x_min y_min width height] format.', trainData.imageFilename(i));
    end
end

% Verify labels are categorical
for i = 1:numRows
    labels = trainingData.labels{i}{1}; % Extract labels for the current image
    if ~iscategorical(labels)
        error('Labels for image %s are not categorical.', trainingData.imageFilename(i));
    end
end

% Verify boxes and labels are cell arrays
if ~iscell(trainingData.boxes) || ~iscell(trainingData.labels)
    error('The boxes and labels columns must be cell arrays.');
end

%% Step 3: Prepare R-CNN Training Pipeline
% Step 1: Extract Unique Classes
% Flatten the nested cell arrays
flatLabels = cellfun(@(x) x{:}, trainingData.labels, 'UniformOutput', false);

% Concatenate all labels into a single categorical array
labels = cat(1, flatLabels{:});

% Get unique classes
uniqueClasses = unique(labels);
numClasses = numel(uniqueClasses); % Number of unique classes

disp(['Number of classes in trainData: ', num2str(numClasses)]);

% Add "background" class if not already present
if ~any(strcmp(uniqueClasses, "Background"))
    uniqueClasses = [uniqueClasses; "Background"];
    numClasses = numClasses + 1; % Increment numClasses to account for background
end

% Load pre-trained ResNet-50
net = resnet50; % Use the pre-trained ResNet-50 model

% Create a layerGraph from the DAGNetwork
lgraph = layerGraph(net);

% Remove unnecessary layers (e.g., 'fc1000' and 'ClassificationLayer_fc1000')
lgraph = removeLayers(lgraph, {'fc1000', 'ClassificationLayer_fc1000', 'fc1000_softmax'});

% Add new layers for object detection
newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'rcnnFC'); % numClasses includes background
lgraph = addLayers(lgraph, newFCLayer);

% Add new softmax layer
softmaxLayer = softmaxLayer('Name', 'rcnnSoftmax'); % Use the built-in function
lgraph = addLayers(lgraph, softmaxLayer);

% Add new classification layer with explicit class names
newClassLayer = classificationLayer('Name', 'rcnnClassification', 'Classes', uniqueClasses);
lgraph = addLayers(lgraph, newClassLayer);

% Connect the layers
lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');
lgraph = connectLayers(lgraph, 'rcnnFC', 'rcnnSoftmax');
lgraph = connectLayers(lgraph, 'rcnnSoftmax', 'rcnnClassification');

% Verify the updated network
disp(lgraph.Layers);
analyzeNetwork(lgraph);

%% Step 3: Visualize Bounding Boxes
% Visualize a few training samples
figure;
numSamples = 4; % Number of samples to visualize
numRows = size(trainingData, 1); % Get number of rows
idx = randperm(numRows, min(numSamples, numRows)); % Ensure it doesn't exceed available rows
for i = 1:numSamples
    subplot(2, 2, i);
    
    % Read the image
    img = imread(trainingData.imageFilename(i));
    
    % Get bounding boxes and labels for the current image
    if isempty(trainingData.boxes{i}) || isempty(trainingData.labels{i})
        error('Bounding boxes or labels for image %s are missing.', trainingData.imageFilename(i));
    end
    bbox = trainingData.boxes{i}{1}; % Extract bounding boxes
    label = trainingData.labels{i}{1}; % Extract labels
    
    % Display the image
    imshow(img);
    hold on;
    
    % Draw bounding boxes and labels
    for j = 1:size(bbox, 1)
        rectangle('Position', bbox(j, :), 'EdgeColor', 'r', 'LineWidth', 2);
        text(bbox(j, 1), bbox(j, 2) - 10, string(label(j)), ...
            'Color', 'red', 'FontSize', 12, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    end
    
    hold off;
end
 
% Define training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 10, ...
    'Verbose', true);
 
%% Step 5: Train the R-CNN Object Detector
% Train the R-CNN detector
detector = trainRCNNObjectDetector(trainingData, lgraph, options, ...
    'NegativeOverlapRange', [0 0.3], ...
    'PositiveOverlapRange', [0.6 1]);
 
disp('? Training Complete! Model successfully trained.');



%% Step 5: Test the Trained Detector
% Test the detector on a sample image
testImage = imread(testingData.imageFilename{1});
[bboxes, scores, labels] = detect(detector, testImage);

if isempty(bboxes)
    warning('No objects detected in the test image.');
else
    % Proceed with visualization
end

% Display the results
figure;
imshow(testImage);
hold on;
for i = 1:size(bboxes, 1)
    rectangle('Position', bboxes(i, :), 'EdgeColor', 'r', 'LineWidth', 2);
    text(bboxes(i, 1), bboxes(i, 2) - 10, string(labels(i)), ...
        'Color', 'red', 'FontSize', 12, 'BackgroundColor', 'white', 'EdgeColor', 'black');
end
hold off;