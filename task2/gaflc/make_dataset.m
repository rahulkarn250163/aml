function make_dataset()
%MAKE_DATASET  Generate the input-output dataset used to score the FLC.
%   No labelled dataset is supplied by the module, so we define a smooth
%   physically motivated target HVAC response and sample it on a grid.
%
%   Target model: a resident's thermally neutral temperature falls as they
%   become more active (an active person prefers a cooler room). The desired
%   HVAC command is a saturating proportional response to the gap between the
%   measured temperature and that neutral point.
%
%   Writes dataset.mat (X [n x 2], y [n x 1]) and dataset.csv.
%   STW7085CEM Task 2, Part 2.

here = fileparts(mfilename('fullpath'));
rng(7);

tg = 10:1:35;            % temperature grid (C)
ag = 0:1:10;             % activity grid
[T,A] = meshgrid(tg,ag);
T = T(:); A = A(:);

Tneutral = 24 - 0.6*A;                       % active -> cooler preference
target   = 100*tanh(-0.14*(T - Tneutral));   % saturating proportional law
noise    = 2.0*randn(size(target));          % mild measurement noise
y        = max(-100, min(100, target + noise));
X        = [T A];

save(fullfile(here,'dataset.mat'),'X','y');
writematrix([X y], fullfile(here,'dataset.csv'));
fprintf('dataset: %d examples; y range [%.1f, %.1f]\n', ...
    numel(y), min(y), max(y));
end
