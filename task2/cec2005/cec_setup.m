function cec_setup()
%CEC_SETUP  Create and store the shift vectors for the CEC'2005 functions.
%   The official CEC'2005 suite shifts each function's global optimum by a
%   fixed vector o so the optimum does not sit at the origin. We generate o
%   once with a fixed seed (so every run is reproducible) and keep it within
%   the inner part of each search domain so the optimum is interior, not on a
%   bound. Vectors are stored up to D = 100 and truncated per dimension.
%
%   F1 Shifted Sphere     : domain [-100,100], shift in [-80,80]
%   F9 Shifted Rastrigin  : domain [-5,5],     shift in [-4,4]
%
%   STW7085CEM Task 2, Part 3.

here = fileparts(mfilename('fullpath'));
rng(2005);
o1 = -80 + 160*rand(1,100);     % sphere shift
o9 =  -4 +   8*rand(1,100);     % rastrigin shift
save(fullfile(here,'cec_shift.mat'),'o1','o9');
fprintf('cec_shift.mat written (o1, o9 of length 100)\n');
end
