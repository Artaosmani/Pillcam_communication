%% plot_PBS.m
%
% Loads PBS results from example_reactive.csv and adds them as markers
% to the current figure (existing analytical curves).
%
% Usage:
%   1. Run analytical script to produce the figure.
%   2. Run this script — it calls hold on and overlays PBS markers.
%
% If no figure is open, it creates a new one with only the PBS markers.
%
% CSV columns produced by example_reactive.cpp:
%   t     : time [s]
%   Pe    : P(diffusing)
%   Pf    : P(bound)
%   Pi    : P(internalized / detected)
%   Pd    : P(degraded)
%   Psum  : Pe+Pf+Pi+Pd  (should be ~1, validation)
%   Pabs  : P(absorbed, perfectly absorbing Rx) — upper bound for Pi

%% Load CSV
csv_file = 'example_reactive.csv';
assert(isfile(csv_file), 'CSV not found. Run ./example_reactive first.');

T = readtable(csv_file);

t_ms  = T.t  * 1e3;   % convert s -> ms for x-axis (matches paper Fig. 2)
Pe    = T.Pe;
Pf    = T.Pf;
Pi    = T.Pi;
Pd    = T.Pd;
Psum  = T.Psum;
Pabs  = T.Pabs;

%% Thin markers — 200 evenly spaced points avoids overplotting
n_markers = 200;
stride    = max(1, floor(length(t_ms) / n_markers));
idx       = 1:stride:length(t_ms);

%% Print Psum check to console
fprintf('PBS Psum validation (should be ~1.0):\n');
fprintf('  t = 0       : %.4f\n', Psum(1));
fprintf('  t = tau/4   : %.4f\n', Psum(round(end/4)));
fprintf('  t = tau/2   : %.4f\n', Psum(round(end/2)));
fprintf('  t = tau     : %.4f\n', Psum(end));

%% Overlay PBS markers on current (or new) figure
hold on;

plot(t_ms(idx), Pe(idx),   'o', 'MarkerSize', 5, ...
    'MarkerEdgeColor', [0.00 0.45 0.74], 'MarkerFaceColor', 'none', ...
    'LineWidth', 1.0, 'DisplayName', 'P_e PBS');

plot(t_ms(idx), Pf(idx),   's', 'MarkerSize', 5, ...
    'MarkerEdgeColor', [0.85 0.33 0.10], 'MarkerFaceColor', 'none', ...
    'LineWidth', 1.0, 'DisplayName', 'P_f PBS');

plot(t_ms(idx), Pi(idx),   '^', 'MarkerSize', 5, ...
    'MarkerEdgeColor', [0.47 0.67 0.19], 'MarkerFaceColor', 'none', ...
    'LineWidth', 1.0, 'DisplayName', 'P_i PBS');

plot(t_ms(idx), Pd(idx),   'd', 'MarkerSize', 5, ...
    'MarkerEdgeColor', [0.93 0.69 0.13], 'MarkerFaceColor', 'none', ...
    'LineWidth', 1.0, 'DisplayName', 'P_d PBS');

plot(t_ms(idx), Pabs(idx), 'v', 'MarkerSize', 5, ...
    'MarkerEdgeColor', [0.30 0.30 0.30], 'MarkerFaceColor', 'none', ...
    'LineWidth', 1.0, 'DisplayName', 'P_{abs} PBS');

legend('show', 'Location', 'east');
