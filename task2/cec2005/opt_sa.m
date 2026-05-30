function [bestVal, curve] = opt_sa(f, lb, ub, maxFES, cp)
%OPT_SA  Simulated annealing (minimisation), hand-coded.
%   Geometric cooling from an initial temperature estimated from the spread
%   of a random sample. Gaussian proposals whose step shrinks with
%   temperature. Metropolis acceptance.
D = numel(lb);
ns = 20;                                   % sample to set initial temperature
Xs = lb + (ub - lb).*rand(ns, D); fs = f(Xs);
T0 = std(fs); if T0 <= 0 || ~isfinite(T0), T0 = 1; end
Tend = T0*1e-4;

x = lb + (ub - lb).*rand(1, D); fx = f(x); fes = 1 + ns;
bestf = fx;
curve = nan(1, numel(cp)); ci = 1;
[curve, ci] = rec(curve, ci, cp, fes, bestf);

sigma0 = 0.10*(ub - lb);
remaining = max(maxFES - fes, 1);
alpha = (Tend/T0)^(1/remaining);
T = T0;
for k = 1:remaining
    sc = 0.001 + 0.999*(T/T0);             % step shrinks as it cools
    xn = x + sc*sigma0.*randn(1, D);
    xn = min(max(xn, lb), ub);
    fn = f(xn); fes = fes + 1;
    if fn < fx || rand < exp(-(fn - fx)/max(T,1e-12))
        x = xn; fx = fn;
    end
    if fx < bestf, bestf = fx; end
    T = T*alpha;
    [curve, ci] = rec(curve, ci, cp, fes, bestf);
end
curve(ci:end) = bestf;
bestVal = bestf;
end

function [curve, ci] = rec(curve, ci, cp, fes, best)
while ci <= numel(cp) && fes >= cp(ci)
    curve(ci) = best; ci = ci + 1;
end
end
