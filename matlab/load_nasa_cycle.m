clc;

% ============================================================
% NASA B0005 - Cycle 2 Discharge Data
% ============================================================

project_root = '/home/rohith/control_systems';
file = fullfile(project_root, 'data', 'raw', 'B0005.mat');;

load(file);

% Select battery
battery = B0005;

% Cycle 2 is the first discharge cycle
cycle = battery.cycle(2);

% Extract measurement data
d = cycle.data;

% Time in seconds
t = double(d.Time(:));

% NASA current is negative during discharge.
% Convert to positive discharge current for the ECM.
I_nasa = -double(d.Current_measured(:));

% Measured terminal voltage
V_nasa = double(d.Voltage_measured(:));

% Temperature
T_nasa = double(d.Temperature_measured(:));

% Capacity
capacity = double(d.Capacity);

% ============================================================
% Simulink From Workspace variables
% ============================================================

I_nasa_ws = [t I_nasa];

V_nasa_ws = [t V_nasa];

T_nasa_ws = [t T_nasa];

% ============================================================
% Display information
% ============================================================

fprintf('============================================\n');
fprintf('NASA B0005 CYCLE 2\n');
fprintf('============================================\n');

fprintf('Samples       : %d\n', length(t));
fprintf('Duration      : %.3f s\n', t(end));
fprintf('Capacity      : %.6f Ah\n', capacity);
fprintf('Current range : %.4f to %.4f A\n', ...
    min(I_nasa), max(I_nasa));
fprintf('Voltage range : %.4f to %.4f V\n', ...
    min(V_nasa), max(V_nasa));
fprintf('Temperature   : %.2f to %.2f C\n', ...
    min(T_nasa), max(T_nasa));

fprintf('============================================\n');
