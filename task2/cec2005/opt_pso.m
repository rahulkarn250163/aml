function [bestVal, curve] = opt_pso(f, lb, ub, maxFES, cp)
%OPT_PSO  Particle swarm optimisation (minimisation), hand-coded.
%   Inertia w = 0.729, cognitive/social c1 = c2 = 1.49445 (Clerc constants),
%   velocity clamped to 20% of the range. Swarm size 40.
D = numel(lb); S = 40;
w = 0.729; c1 = 1.49445; c2 = 1.49445;
X = lb + (ub - lb).*rand(S, D);
V = zeros(S, D);
vmax = 0.20*(ub - lb);
fit = f(X); fes = S;
pbest = X; pbestf = fit;
[gbestf, gi] = min(fit); gbest = X(gi, :);
curve = nan(1, numel(cp)); ci = 1;
[curve, ci] = rec(curve, ci, cp, fes, gbestf);

while fes < maxFES
    r1 = rand(S, D); r2 = rand(S, D);
    V = w*V + c1*r1.*(pbest - X) + c2*r2.*(gbest - X);
    V = min(max(V, -vmax), vmax);
    X = X + V;
    X = min(max(X, lb), ub);
    fit = f(X); fes = fes + S;
    imp = fit < pbestf;
    pbest(imp, :) = X(imp, :); pbestf(imp) = fit(imp);
    [bv, bi] = min(pbestf);
    if bv < gbestf, gbestf = bv; gbest = pbest(bi, :); end
    [curve, ci] = rec(curve, ci, cp, fes, gbestf);
end
curve(ci:end) = gbestf;
bestVal = gbestf;
end

function [curve, ci] = rec(curve, ci, cp, fes, best)
while ci <= numel(cp) && fes >= cp(ci)
    curve(ci) = best; ci = ci + 1;
end
end
