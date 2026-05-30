function run_all()
%RUN_ALL  Reproduce every Task 2 result, figure and table end to end.
%   Usage:  matlab -batch "run_all"
%
%   STW7085CEM Task 2 (Fuzzy Logic Controller + GA + CEC'2005 comparison).

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'flc'), fullfile(here,'gaflc'), fullfile(here,'cec2005'));

fprintf('==== Part 1: Fuzzy Logic Controller ====\n');
save_flc();
flc_evidence();

fprintf('==== Part 2: GA optimisation of the FLC ====\n');
make_dataset();
ga_optimise_flc();

fprintf('==== Part 3: CEC2005 optimiser comparison ====\n');
cec_setup();
cec_landscapes();
compare_cec();
compare_cec_toolbox();

fprintf('==== Task 2 pipeline complete ====\n');
end
