%% Initialize data.mat as a cell array if it doesn't exist or is not a cell array.
if exist('data.mat', 'file') == 2
    load('data.mat', 'data');
    if ~iscell(data)
        data = {};
    end
else
    data = {};
end
save('data.mat', 'data');

%% New parameter ranges:
fault_location_values = 0.2:0.2:3;         % Fault location from 0 to 3 (step 0.2)
fault_resistance_values = 0:0.25:2;          % Fault resistance from 0 to 2 (step 0.25)
P_LOAD_DC_values = 4.5:0.2:5.5;              % P_LOAD_DC from 4.5 to 5.5 (step 0.2)

iterationCount = 0;
numReps = 5;  % Number of repetitions for each set of parameters

for rep = 1:numReps
    for fault_location = fault_location_values
        for fault_resistance = fault_resistance_values
            for P_LOAD_DC = P_LOAD_DC_values
                iterationCount = iterationCount + 1;
                
                %% --- Generate random fault timing parameters ---
                % Fault onset time: uniformly from 0.1 sec to 0.4 sec.
                fault_time = 0.1 + (0.4 - 0.1) * rand();
                
                % Fault duration (in percentage of simulation time, where simulation time is 0.5 sec)
                % For transient faults: duration between 0.05 and 0.15 sec (i.e., 10% to 30%)
                % For persistent faults: duration between 0.15 and 0.3 sec (i.e., 30% to 60%)
                if rand() < 0.5
                    fault_duration = 10 + (30 - 10) * rand(); % transient fault duration in percent
                else
                    fault_duration = 30 + (60 - 30) * rand();   % persistent fault duration in percent
                end
                
                % Calculate R_LOAD_DC based on P_LOAD_DC.
                R_LOAD_DC = 90 / P_LOAD_DC;
                
                %% --- Assign fault parameters to the base workspace ---
                assignin('base', 'fault_time', fault_time);
                assignin('base', 'fault_duration', fault_duration);
                assignin('base', 'fault_location', fault_location);
                assignin('base', 'fault_resistance', fault_resistance);
                assignin('base', 'R_LOAD_DC', R_LOAD_DC);
                
                %% --- Run simulation ---
                simOut = sim('Model_DC_MG_Main.slx');
                
                %% --- Load simulation outputs and organize data ---
                load('V_B.mat', 'Vb');      % Vb is assumed to be 2×N (e.g., 2×600)
                load('I_B.mat', 'Ib');      % Ib is assumed to be 2×N
                load('labels.mat', 'label');% label is assumed to be 2×N
                
                commonRow = Vb(1, :);
                VbUnique  = Vb(2, :);
                IbUnique  = Ib(2, :);
                labelUnique = label(2, :);
                
                % Create a cell array for this iteration that includes fault parameters.
                iterationData = {commonRow(:)'; VbUnique(:)'; IbUnique(:)'; labelUnique(:)'; fault_time; fault_duration; fault_location; fault_resistance; R_LOAD_DC};

                % Ensure iterationData is a 1×9 cell array.
                iterationData = reshape(iterationData, 1, []);
                
                load('data.mat', 'data');
                data = [data; iterationData];
                save('data.mat', 'data');
                
                %% --- Display progress ---
                fprintf('Iteration %d: fault_time = %.3f sec, fault_duration = %.3f%%, fault_location = %.2f, fault_resistance = %.2f, R_LOAD_DC = %.2f\n', ...
                    iterationCount, fault_time, fault_duration, fault_location, fault_resistance, R_LOAD_DC);
            end
        end
    end
end
