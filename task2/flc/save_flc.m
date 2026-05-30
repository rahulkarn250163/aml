function save_flc()
%SAVE_FLC  Build the FLC and write it to room_flc.fis next to this file.
here = fileparts(mfilename('fullpath'));
fis  = build_flc();
writeFIS(fis, fullfile(here,'room_flc.fis'));
fprintf('Wrote %s\n', fullfile(here,'room_flc.fis'));

% Quick smoke test: a few representative operating points.
tests = [32 8;   % hot and active     -> strong cooling
         22 2;   % comfortable, resting-> near off
         12 2;   % cold and resting   -> strong heating
         26 5];  % warm, medium       -> cooling
for i = 1:size(tests,1)
    y = evalfis(fis, tests(i,:));
    fprintf('temp=%4.1f activity=%4.1f -> hvac=%7.2f\n', ...
        tests(i,1), tests(i,2), y);
end
end
