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
    
    %AB
    classNames = replace(classNames, " ", "_");

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
    % @ABAssume classNames is an Mx1 string array
    M = numel(classNames);
    
    % Define variable names: 'imageFilename' followed by class names
    variableNames = ["imageFilename", classNames']; % Convert to row for concatenation
    % Define variable types: first column as 'string', the rest as 'cell'
    variableTypes = ["string", repmat("cell", 1, M)];
    % Create the table
    data = table('Size', [numel(imagePaths), M+1], ...
             'VariableTypes', cellstr(variableTypes), ...
             'VariableNames', cellstr(variableNames));
    
    % @AB OLD data = table('Size', [numel(imagePaths), 3], ...
    %              'VariableTypes', {'string', 'cell', 'cell'}, ... % Ensure 'boxes' and 'labels' are cell arrays
    %              'VariableNames', {'imageFilename', 'boxes', 'labels'});

    % Parse annotations and populate the table
    for i = 1:numel(imagePaths)
        annotation = annotations{i};
        annotationLines = splitlines(annotation); % Split into individual lines

        % @AB Initialize arrays to store bounding boxes and labels for this image
        %boxes = [];
        %labels = [];

        boxes = cell(1,M);
        %labels = cell(1,M);
        
        
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
            boxes{class_id} = [boxes{class_id}; x_min y_min width * imageSize(2) height * imageSize(1)];

            % Add the corresponding label
         %   labels{class} = [labels; classNames(class_id)];
        end
        
        % Debug: Display labels for the current image
   %     disp(['Labels for image ', num2str(i), ':']);
    %    disp(labels);

        % Add the data to the table
        data.imageFilename(i) = imagePaths{i};
        %data.boxes{i} = {boxes}; % Wrap in a cell array
        for jj=1:M
            data.(classNames{jj}){i}=boxes{jj};
        end    
        %data.boxes{i} = {boxes}; % Wrap in a cell array
        %data.labels{i} = {categorical(labels, classNames)}; % Wrap in a cell array
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
    flag_empty=true;
%    if isempty(trainingData.boxes{i})
      for jj=1:M
            if ~isempty(trainingData.(classNames{jj}){i})
             flag_empty=false;   
             break;
            end
        end  
    
     %if isempty(trainingData.boxes{i})
     if flag_empty
        error('Bounding boxes for image %s are missing.', trainingData.imageFilename(i));
    end
%    bbox = trainingData.boxes{i}{1}; % Extract bounding boxes for the current image
    %if size(bbox, 2) ~= 4
    %    error('Bounding boxes for image %s are not in [x_min y_min width height] format.', trainData.imageFilename(i));
    %end
end



% Verify labels are categorical
%for i = 1:numRows
%    labels = trainingData.labels{i}{1}; % Extract labels for the current image
%    if ~iscategorical(labels)
%        error('Labels for image %s are not categorical.', trainingData.imageFilename(i));
%    end
%end

% Verify boxes and labels are cell arrays
%if ~iscell(trainingData.boxes) || ~iscell(trainingData.labels)
%    error('The boxes and labels columns must be cell arrays.');
%end

%% Step 3: Prepare R-CNN Training Pipeline
% Step 1: Extract Unique Classes
% Flatten the nested cell arrays
%flatLabels = cellfun(@(x) x{:}, trainingData.labels, 'UniformOutput', false);

% Concatenate all labels into a single categorical array
%labels = cat(1, flatLabels{:});

% Get unique classes
%uniqueClasses = unique(labels);
%numClasses = numel(uniqueClasses); % Number of unique classes

uniqueClasses=classNames;
numClasses = numel(uniqueClasses);
%disp(['Number of classes in trainData: ', num2str(numClasses)]);

% Add "background" class if not already present
if ~any(strcmp(uniqueClasses, "Background"))
    uniqueClasses = [uniqueClasses; "Background"];
    numClasses = numClasses + 1; % Increment numClasses to account for background
end

uniqueClasses
numClasses


% Load trained Faster R-CNN model
%load('trainedRCNNDetector.mat', 'trainedDetector');

% Use testingData prepared earlier
numImages = size(testingData, 1);
classNames = classNames(:); % Ensure column vector

% Initialize counters
TP = 0; FP = 0; FN = 0;

% IoU threshold for a match
iouThreshold = 0.5;

for i = 1:numImages
    img = imread(testingData.imageFilename(i));
    [bboxes, scores, labels] = detect(detector, img);

    % Gather ground truth bboxes from class-wise columns
    gtBoxes = [];
    for j = 1:numel(classNames)
        classBoxes = testingData.(classNames(j)){i};
        if ~isempty(classBoxes)
            gtBoxes = [gtBoxes; classBoxes];
        end
    end

    % If no detections and no ground truth, continue
    if isempty(bboxes) && isempty(gtBoxes)
        continue;
    end

    % Match detected boxes to ground truth using IoU
    matchedGT = false(size(gtBoxes, 1), 1);
    for d = 1:size(bboxes, 1)
        maxIoU = 0;
        matchIdx = 0;
        for g = 1:size(gtBoxes, 1)
            iou = bboxOverlapRatio(bboxes(d,:), gtBoxes(g,:));
            if iou > maxIoU
                maxIoU = iou;
                matchIdx = g;
            end
        end

        if maxIoU >= iouThreshold && ~matchedGT(matchIdx)
            TP = TP + 1;
            matchedGT(matchIdx) = true;
        else
            FP = FP + 1;
        end
    end

    % Count unmatched ground truth boxes as FN
    FN = FN + sum(~matchedGT);
end

% Compute accuracy
accuracy = TP / (TP + FP + FN);
disp(['Accuracy: ', num2str(accuracy)]);
