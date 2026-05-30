function P = cec_problem(name, D)
%CEC_PROBLEM  Return a CEC'2005 benchmark problem definition.
%   P = cec_problem('F1', D) or cec_problem('F9', D)
%
%   Fields: f (objective handle, accepts n-by-D, returns n-by-1),
%           lb, ub (1-by-D bounds), fopt (known optimum value),
%           D, name, o (shift vector).
%
%   F1 Shifted Sphere     f(x) = sum((x-o).^2) + f_bias,  f_bias = -450
%   F9 Shifted Rastrigin  f(x) = sum(z.^2 - 10cos(2*pi*z) + 10) + f_bias,
%                          z = x-o, f_bias = -330
%
%   STW7085CEM Task 2, Part 3.

here = fileparts(mfilename('fullpath'));
if ~exist(fullfile(here,'cec_shift.mat'),'file'); cec_setup(); end
S = load(fullfile(here,'cec_shift.mat'));

switch upper(name)
    case 'F1'
        o = S.o1(1:D); bias = -450; lo = -100; hi = 100;
        P.f = @(x) sum((x - o).^2, 2) + bias;
        P.name = 'F1 Shifted Sphere';
    case 'F9'
        o = S.o9(1:D); bias = -330; lo = -5; hi = 5;
        P.f = @(x) sum((x - o).^2 - 10*cos(2*pi*(x - o)) + 10, 2) + bias;
        P.name = 'F9 Shifted Rastrigin';
    otherwise
        error('Unknown function %s', name);
end
P.lb = lo*ones(1,D);
P.ub = hi*ones(1,D);
P.fopt = bias;
P.D = D;
P.o = o;
end
