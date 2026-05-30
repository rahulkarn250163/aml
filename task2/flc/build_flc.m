function fis = build_flc()
%BUILD_FLC  Mamdani fuzzy logic controller for climate and comfort control
%   in one room of an intelligent assistive-care flat.
%
%   Inputs : temperature (10..35 C), activity (0..10 activity index)
%   Output : hvac actuator (-100 full cooling .. +100 full heating)
%
%   Inference : Mamdani, AND = min, aggregation = max, implication = min,
%               defuzzification = centroid.
%
%   Returns a mamfis object. Also written to room_flc.fis by save_flc.m.
%
%   STW7085CEM Task 2, Part 1.

fis = mamfis( ...
    'Name','RoomClimateFLC', ...
    'AndMethod','min', ...
    'OrMethod','max', ...
    'ImplicationMethod','min', ...
    'AggregationMethod','max', ...
    'DefuzzificationMethod','centroid');

% ---------------------------------------------------------------- inputs
% Temperature: five sets across a comfort-relevant range.
fis = addInput(fis,[10 35],'Name','temperature');
fis = addMF(fis,'temperature','trapmf',[10 10 13 17],'Name','Cold');
fis = addMF(fis,'temperature','trimf', [15 19 22],   'Name','Cool');
fis = addMF(fis,'temperature','trimf', [20 23 26],   'Name','Comfortable');
fis = addMF(fis,'temperature','trimf', [24 27 30],   'Name','Warm');
fis = addMF(fis,'temperature','trapmf',[28 32 35 35],'Name','Hot');

% Activity / occupancy level: three sets.
fis = addInput(fis,[0 10],'Name','activity');
fis = addMF(fis,'activity','trapmf',[0 0 2 4],   'Name','Low');
fis = addMF(fis,'activity','trimf', [3 5 7],     'Name','Medium');
fis = addMF(fis,'activity','trapmf',[6 8 10 10], 'Name','High');

% ---------------------------------------------------------------- output
% HVAC actuator: negative cools, positive heats.
fis = addOutput(fis,[-100 100],'Name','hvac');
fis = addMF(fis,'hvac','trimf',[-100 -100 -50],'Name','StrongCool');
fis = addMF(fis,'hvac','trimf',[-75 -40 -5],   'Name','Cool');
fis = addMF(fis,'hvac','trimf',[-20 0 20],     'Name','Off');
fis = addMF(fis,'hvac','trimf',[5 40 75],      'Name','Heat');
fis = addMF(fis,'hvac','trimf',[50 100 100],   'Name','StrongHeat');

% ---------------------------------------------------------------- rules
% Rule matrix rows = temperature (Cold..Hot), cols = activity (Low/Med/High).
% Higher activity adds body heat, so the controller biases cooler.
% Columns of ruleList: [temp activity hvac weight connective(1=AND)].
ruleList = [ ...
    1 1 5 1 1;   % Cold        & Low    -> StrongHeat
    1 2 5 1 1;   % Cold        & Medium -> StrongHeat
    1 3 4 1 1;   % Cold        & High   -> Heat
    2 1 4 1 1;   % Cool        & Low    -> Heat
    2 2 4 1 1;   % Cool        & Medium -> Heat
    2 3 3 1 1;   % Cool        & High   -> Off
    3 1 3 1 1;   % Comfortable & Low    -> Off
    3 2 3 1 1;   % Comfortable & Medium -> Off
    3 3 2 1 1;   % Comfortable & High   -> Cool
    4 1 2 1 1;   % Warm        & Low    -> Cool
    4 2 2 1 1;   % Warm        & Medium -> Cool
    4 3 1 1 1;   % Warm        & High   -> StrongCool
    5 1 1 1 1;   % Hot         & Low    -> StrongCool
    5 2 1 1 1;   % Hot         & Medium -> StrongCool
    5 3 1 1 1];  % Hot         & High   -> StrongCool
fis = addRule(fis,ruleList);
end
