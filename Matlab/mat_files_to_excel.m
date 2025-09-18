%% Load data from data.mat
load('data.mat', 'data');  % data is assumed to be an M×9 cell array.
numIter = size(data,1);
numDataPoints = size(data{1,1}, 2);  % number of columns in the time series (commonRow)

% Total columns = 2 (Iteration and Type) + numDataPoints (time series data) + 5 fault parameter columns.
totalCols = 2 + numDataPoints + 5;

%% Create an output cell array with header row.
outData = cell(1 + 4*numIter, totalCols);

% Create header row
header = cell(1, totalCols);
header{1} = 'Iteration';
header{2} = 'Type';
% Time series headers
for j = 1:numDataPoints
    header{2+j} = sprintf('C%d', j);
end
% Fault parameter headers (last 5 columns)
header{end-4} = 'fault_time';
header{end-3} = 'fault_duration';
header{end-2} = 'fault_location';
header{end-1} = 'fault_resistance';
header{end}   = 'R_LOAD_DC';

outData(1,:) = header;

%% Fill in data for each iteration.
for i = 1:numIter
    % Determine starting row for iteration i (accounting for header row)
    startRow = 1 + (i-1)*4 + 1;
    
    % Extract fault parameters from the current iteration data.
    ft   = data{i,5};  % fault_time
    fd   = data{i,6};  % fault_duration (in %)
    floc = data{i,7};  % fault_location
    fres = data{i,8};  % fault_resistance
    R_DC = data{i,9};  % R_LOAD_DC
    
    % Create a cell array for fault parameters (same for all 4 rows)
    faultParams = {ft, fd, floc, fres, R_DC};
    
    % Row for time (common row is stored in data{i,1})
    outData{startRow, 1} = i;
    outData{startRow, 2} = 'time';
    outData(startRow, 3:2+numDataPoints) = num2cell(data{i,1});
    outData(startRow, 3+numDataPoints:end) = faultParams;
    
    % Row for voltage (data{i,2})
    outData{startRow+1, 1} = i;
    outData{startRow+1, 2} = 'voltage';
    outData(startRow+1, 3:2+numDataPoints) = num2cell(data{i,2});
    outData(startRow+1, 3+numDataPoints:end) = faultParams;
    
    % Row for current (data{i,3})
    outData{startRow+2, 1} = i;
    outData{startRow+2, 2} = 'current';
    outData(startRow+2, 3:2+numDataPoints) = num2cell(data{i,3});
    outData(startRow+2, 3+numDataPoints:end) = faultParams;
    
    % Row for label (data{i,4})
    outData{startRow+3, 1} = i;
    outData{startRow+3, 2} = 'label';
    outData(startRow+3, 3:2+numDataPoints) = num2cell(data{i,4});
    outData(startRow+3, 3+numDataPoints:end) = faultParams;
end

%% Write the output cell array to an Excel file.
writecell(outData, 'data_detailed.xlsx');
fprintf('Data saved to data_detailed.xlsx\n');
