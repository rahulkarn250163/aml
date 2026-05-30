function [bestVal, curve] = opt_ga(f, lb, ub, maxFES, cp)
%OPT_GA  Real-coded genetic algorithm (minimisation), hand-coded.
%   Tournament selection (size 3), BLX-0.5 crossover, Gaussian mutation
%   (rate 0.10), elitism (best 2). Population 50.
%   Returns best objective value and the best-so-far value at the function
%   evaluation checkpoints cp.
D = numel(lb); N = 50;
P   = lb + (ub - lb).*rand(N, D);
fit = f(P); fes = N;
bestVal = min(fit);
curve = nan(1, numel(cp)); ci = 1;
[curve, ci] = rec(curve, ci, cp, fes, bestVal);
sigma0 = 0.10*(ub - lb);

while fes < maxFES
    [fit, ord] = sort(fit); P = P(ord, :);
    newP = P(1:2, :);                      % elitism
    while size(newP,1) < N
        a = tour(fit); b = tour(fit);
        c = blx(P(a,:), P(b,:), 0.5);
        m = rand(1,D) < 0.10;
        if any(m), c(m) = c(m) + sigma0(m).*randn(1,sum(m)); end
        c = min(max(c, lb), ub);
        newP(end+1,:) = c; %#ok<AGROW>
    end
    P = newP(1:N, :);
    fit = f(P); fes = fes + N;
    bv = min(fit); if bv < bestVal, bestVal = bv; end
    [curve, ci] = rec(curve, ci, cp, fes, bestVal);
end
curve(ci:end) = bestVal;
end

function a = tour(fit)
n = numel(fit); i = randi(n, 3, 1); [~, k] = min(fit(i)); a = i(k);
end

function c = blx(p, q, al)
mn = min(p,q); mx = max(p,q); I = mx - mn;
lo = mn - al*I; hi = mx + al*I;
c = lo + (hi - lo).*rand(1, numel(p));
end

function [curve, ci] = rec(curve, ci, cp, fes, best)
while ci <= numel(cp) && fes >= cp(ci)
    curve(ci) = best; ci = ci + 1;
end
end
