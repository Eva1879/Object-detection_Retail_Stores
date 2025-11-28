% Set your annotation folder
folder = 'C:\Users\grace\Downloads\dataset\train';  % <-- Change this
files = dir(fullfile(folder, '*.txt'));

imageSize = 608;  % Assuming all images are 608x608
classMap = containers.Map('KeyType', 'int32', 'ValueType', 'any');

for k = 1:length(files)
    filePath = fullfile(folder, files(k).name);
    data = readmatrix(filePath);

    % Handle single-row files
    if isvector(data) && numel(data) == 5
        data = reshape(data, 1, []);
    end

    if isempty(data)
        continue;
    end

    for i = 1:size(data, 1)
        class_id = int32(data(i, 1));
        x_center = data(i, 2) * imageSize;
        y_center = data(i, 3) * imageSize;
        width    = data(i, 4) * imageSize;
        height   = data(i, 5) * imageSize;

        x_min = x_center - width / 2;
        y_min = y_center - height / 2;
        x_max = x_center + width / 2;
        y_max = y_center + height / 2;

        bbox = [x_min, y_min, x_max, y_max, width, height];

        if isKey(classMap, class_id)
            classMap(class_id) = [classMap(class_id); bbox];
        else
            classMap(class_id) = bbox;
        end
    end
end

% Print per-class stats including range
fprintf('\nPixelized Bounding Box Stats per Class (YOLO format)\n');
fprintf('====================================================================\n');

classKeys = keys(classMap);
for i = 1:length(classKeys)
    cid = classKeys{i};
    boxes = classMap(cid);

    x_min_vals = boxes(:, 1);
    y_min_vals = boxes(:, 2);
    x_max_vals = boxes(:, 3);
    y_max_vals = boxes(:, 4);
    widths     = boxes(:, 5);
    heights    = boxes(:, 6);

    fprintf('Class %d:\n', cid);
    fprintf('  x_min:   min = %.2f, max = %.2f, mean = %.2f, range = %.2f\n', ...
        min(x_min_vals), max(x_min_vals), mean(x_min_vals), range(x_min_vals));
    fprintf('  x_max:   min = %.2f, max = %.2f, mean = %.2f, range = %.2f\n', ...
        min(x_max_vals), max(x_max_vals), mean(x_max_vals), range(x_max_vals));
    fprintf('  y_min:   min = %.2f, max = %.2f, mean = %.2f, range = %.2f\n', ...
        min(y_min_vals), max(y_min_vals), mean(y_min_vals), range(y_min_vals));
    fprintf('  y_max:   min = %.2f, max = %.2f, mean = %.2f, range = %.2f\n', ...
        min(y_max_vals), max(y_max_vals), mean(y_max_vals), range(y_max_vals));
    fprintf('  width:   min = %.2f, max = %.2f, mean = %.2f, range = %.2f\n', ...
        min(widths), max(widths), mean(widths), range(widths));
    fprintf('  height:  min = %.2f, max = %.2f, mean = %.2f, range = %.2f\n', ...
        min(heights), max(heights), mean(heights), range(heights));
    fprintf('--------------------------------------------------------------------\n');
end
