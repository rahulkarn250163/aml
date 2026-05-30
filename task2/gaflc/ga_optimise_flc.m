function ga_optimise_flc()
%GA_OPTIMISE_FLC  Tune the FLC membership functions with a hand-coded GA.
%
%   Encoding : real-valued chromosome of membership-function breakpoints
%              (see flc_template). Chromosome length is reported below.
%   Selection: tournament (size 3).
%   Crossover: BLX-alpha (blend), rate 0.9.
%   Mutation : Gaussian, per-gene, rate 0.15, sigma = 0.10 * range.
%   Repair   : clamp to range + sort breakpoints (in flc_decode).
%   Fitness  : minimise MSE between FLC output and the target dataset.
%   Elitism  : best 2 carried over each generation.
%
%   Also runs MATLAB's ga() as an independent cross-check.
%   Saves figures to ../figures and metrics to ../results.
%   STW7085CEM Task 2, Part 2.

here   = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'..','flc'));
figdir = fullfile(here,'..','figures');
resdir = fullfile(here,'..','results');
if ~exist(figdir,'dir'); mkdir(figdir); end
if ~exist(resdir,'dir'); mkdir(resdir); end
set(groot,'defaultFigureColor','w');
% Suppress the expected "no rules fired" warning for candidates whose tuned
% partitions leave a temporary coverage gap; such candidates are penalised
% through their MSE anyway.
warning('off','fuzzy:general:evalfis:NoRuleFired');
warning('off','MATLAB:fuzzy:evalfis:NoRuleFired');
ws = warning('off','all');

% ---- dataset ----
if ~exist(fullfile(here,'dataset.mat'),'file'); make_dataset(); end
S = load(fullfile(here,'dataset.mat')); X = S.X; y = S.y;

% ---- problem setup ----
tmpl = flc_template();
L  = tmpl.L; lb = tmpl.lb; ub = tmpl.ub; x0 = tmpl.x0;
baseMSE = flc_fitness(x0, tmpl, X, y);
fprintf('Chromosome length L = %d\n', L);
fprintf('Baseline (hand-designed) MSE = %.4f  RMSE = %.4f\n', ...
    baseMSE, sqrt(baseMSE));

% ---- GA hyperparameters ----
rng(42);
pop = 50; gen = 80; tour = 3; pc = 0.9; alpha = 0.5; pm = 0.15; elite = 2;
sigma = 0.10 * (ub - lb);

% ---- initialise population (seed one individual with the hand design) ----
P = zeros(pop, L);
P(1,:) = x0';
for i = 2:pop
    if i <= ceil(pop/2)
        P(i,:) = (x0 + (0.15*(ub-lb)).*randn(L,1))';   % perturb the design
    else
        P(i,:) = (lb + (ub-lb).*rand(L,1))';           % random restart
    end
end
P   = min(max(P, lb'), ub');
fit = arrayfit(P, tmpl, X, y);

% ---- evolution loop ----
hist = zeros(gen,1);
for gix = 1:gen
    [fit, ord] = sort(fit);
    P = P(ord,:);
    hist(gix) = fit(1);
    newP = P(1:elite,:);
    while size(newP,1) < pop
        p1 = tournament(P, fit, tour);
        p2 = tournament(P, fit, tour);
        if rand < pc
            c = blx(p1, p2, alpha);
        else
            c = p1;
        end
        c = gauss_mut(c, pm, sigma, lb, ub);
        newP(end+1,:) = c'; %#ok<AGROW>
    end
    P   = newP(1:pop,:);
    fit = arrayfit(P, tmpl, X, y);
    if mod(gix,10)==0 || gix==1
        fprintf('  gen %3d  best MSE %.4f\n', gix, min(fit));
    end
end
[gaMSE, bi] = min(fit);
xbest = P(bi,:)';
fprintf('Hand-coded GA best MSE = %.4f  RMSE = %.4f\n', gaMSE, sqrt(gaMSE));

% ---- save optimised FIS ----
fisOpt = flc_decode(xbest, tmpl);
writeFIS(fisOpt, fullfile(here,'room_flc_opt.fis'));
save(fullfile(here,'ga_solution.mat'),'xbest','hist','baseMSE','gaMSE','L');

% ---- toolbox ga() cross-check ----
gaTB = NaN;
if license('test','GADS_Toolbox') && ~isempty(which('ga')) && ...
        contains(lower(which('ga')),'globaloptim')
    opts = optimoptions('ga','PopulationSize',pop, ...
        'MaxGenerations',gen,'Display','off');
    [~, gaTB] = ga(@(z) flc_fitness(z(:), tmpl, X, y), ...
        L, [],[],[],[], lb, ub, [], opts);
    fprintf('Toolbox ga() best MSE = %.4f  RMSE = %.4f\n', gaTB, sqrt(gaTB));
else
    fprintf('Toolbox ga() not available, cross-check skipped.\n');
end

% ---- predictions before/after ----
yb = evalfis(tmpl.base, X);
ya = evalfis(fisOpt,    X);

% ================= figures =================
% convergence
f = figure('Theme','light','Position',[100 100 760 460]);
plot(1:gen, hist, 'LineWidth',1.8); hold on;
yline(baseMSE,'r--','baseline (hand design)','LineWidth',1.3);
if isfinite(gaTB)
    yline(gaTB,'g-.','toolbox ga()','LineWidth',1.3);
end
xlabel('generation'); ylabel('best MSE'); grid on;
title('GA convergence: FLC tuning');
exportgraphics(f, fullfile(figdir,'p2_convergence.png'),'Resolution',160);
close(f);

% control surface before/after
f = figure('Theme','light','Position',[100 100 760 580]);
gensurf(tmpl.base); title('Control surface before GA'); view(135,30);
exportgraphics(f, fullfile(figdir,'p2_surface_before.png'),'Resolution',160);
close(f);
f = figure('Theme','light','Position',[100 100 760 580]);
gensurf(fisOpt); title('Control surface after GA'); view(135,30);
exportgraphics(f, fullfile(figdir,'p2_surface_after.png'),'Resolution',160);
close(f);

% predicted vs target scatter
f = figure('Theme','light','Position',[100 100 760 460]);
plot(y, yb, '.', 'MarkerSize',8); hold on;
plot(y, ya, '.', 'MarkerSize',8);
plot([-100 100],[-100 100],'k:');
xlabel('target hvac'); ylabel('FLC output'); grid on; axis equal;
xlim([-100 100]); ylim([-100 100]);
legend({sprintf('before (RMSE %.1f)',sqrt(baseMSE)), ...
        sprintf('after  (RMSE %.1f)',sqrt(gaMSE))},'Location','northwest');
title('FLC output vs target, before and after GA');
exportgraphics(f, fullfile(figdir,'p2_pred_vs_target.png'),'Resolution',160);
close(f);

% membership functions before/after (temperature and hvac)
plot_mf_ba(tmpl.base, fisOpt, 'input', 1, 'temperature', ...
    fullfile(figdir,'p2_mf_temperature_ba.png'));
plot_mf_ba(tmpl.base, fisOpt, 'output', 1, 'hvac', ...
    fullfile(figdir,'p2_mf_hvac_ba.png'));

% ================= metrics =================
fid = fopen(fullfile(resdir,'p2_results.txt'),'w');
fprintf(fid,'Part 2 - GA optimisation of the FLC\n');
fprintf(fid,'Chromosome length L = %d (real-valued MF breakpoints)\n', L);
fprintf(fid,'GA: pop=%d gen=%d tournament=%d pc=%.2f BLX-alpha=%.2f pm=%.2f elite=%d\n', ...
    pop,gen,tour,pc,alpha,pm,elite);
fprintf(fid,'Baseline MSE = %.4f  (RMSE %.4f)\n', baseMSE, sqrt(baseMSE));
fprintf(fid,'Hand-coded GA MSE = %.4f  (RMSE %.4f)\n', gaMSE, sqrt(gaMSE));
if isfinite(gaTB)
    fprintf(fid,'Toolbox ga() MSE  = %.4f  (RMSE %.4f)\n', gaTB, sqrt(gaTB));
end
fprintf(fid,'Improvement = %.1f%% reduction in MSE\n', 100*(baseMSE-gaMSE)/baseMSE);
fclose(fid);
type(fullfile(resdir,'p2_results.txt'));

T = table(L, baseMSE, gaMSE, gaTB, 100*(baseMSE-gaMSE)/baseMSE, ...
    'VariableNames', {'L','baseMSE','gaMSE','toolboxMSE','pctImprove'});
writetable(T, fullfile(resdir,'p2_metrics.csv'));
fprintf('\nPart 2 done.\n');
end


% ----------------------------------------------------------- helpers
function f = arrayfit(P, tmpl, X, y)
n = size(P,1); f = zeros(n,1);
for i = 1:n
    f(i) = flc_fitness(P(i,:)', tmpl, X, y);
end
end

function ind = tournament(P, fit, k)
n = size(P,1);
idx = randi(n, k, 1);
[~, b] = min(fit(idx));
ind = P(idx(b),:)';
end

function c = blx(p1, p2, alpha)
p1 = p1(:); p2 = p2(:);
cmin = min(p1,p2); cmax = max(p1,p2); I = cmax - cmin;
lo = cmin - alpha*I; hi = cmax + alpha*I;
c = lo + (hi-lo).*rand(numel(p1),1);
end

function c = gauss_mut(c, pm, sigma, lb, ub)
c = c(:);
m = rand(numel(c),1) < pm;
c(m) = c(m) + sigma(m).*randn(sum(m),1);
c = min(max(c, lb), ub);
end

function plot_mf_ba(fisB, fisA, kind, idx, name, outfile)
set(groot,'defaultFigureColor','w');
f = figure('Theme','light','Position',[100 100 820 360]);
subplot(1,2,1); plotmf(fisB, kind, idx);
title([name ' before']); xlabel(name); ylabel('degree'); grid on;
subplot(1,2,2); plotmf(fisA, kind, idx);
title([name ' after']);  xlabel(name); ylabel('degree'); grid on;
exportgraphics(f, outfile, 'Resolution',160);
close(f);
end
