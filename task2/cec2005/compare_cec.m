function compare_cec()
%COMPARE_CEC  Compare GA, PSO and SA on two CEC'2005 functions.
%   Functions : F1 Shifted Sphere (unimodal), F9 Shifted Rastrigin (multimodal)
%   Dimensions: D = 2 and D = 10
%   Budget    : maxFES = 10000 * D function evaluations per run
%   Runs      : 15 independent runs per (function, dimension, optimiser)
%   Reports   : mean, std, best and worst final error (f - f*) and averaged
%               best-so-far convergence curves.
%
%   STW7085CEM Task 2, Part 3.

here   = fileparts(mfilename('fullpath'));
figdir = fullfile(here,'..','figures');
resdir = fullfile(here,'..','results');
if ~exist(figdir,'dir'); mkdir(figdir); end
if ~exist(resdir,'dir'); mkdir(resdir); end
if ~exist(fullfile(here,'cec_shift.mat'),'file'); cec_setup(); end
set(groot,'defaultFigureColor','w');

funcs = {'F1','F9'};
Ds    = [2 10];
nRun  = 15;
opt   = {'GA',@opt_ga; 'PSO',@opt_pso; 'SA',@opt_sa};
nOpt  = size(opt,1);

rows = cell(0,7);
t0 = tic;
for fi = 1:numel(funcs)
    for di = 1:numel(Ds)
        D = Ds(di);
        Pr = cec_problem(funcs{fi}, D);
        maxFES = 10000*D;
        cp = unique(round(logspace(log10(50), log10(maxFES), 60)));

        finalErr = zeros(nRun, nOpt);
        curves   = cell(1, nOpt);
        for o = 1:nOpt, curves{o} = zeros(nRun, numel(cp)); end

        for o = 1:nOpt
            for r = 1:nRun
                rng(1000*o + r);
                [bv, cv] = opt{o,2}(Pr.f, Pr.lb, Pr.ub, maxFES, cp);
                finalErr(r,o)  = bv - Pr.fopt;
                curves{o}(r,:) = cv - Pr.fopt;
            end
            e = finalErr(:,o);
            rows(end+1,:) = {Pr.name, D, opt{o,1}, ...
                mean(e), std(e), min(e), max(e)}; %#ok<AGROW>
            fprintf('%-22s D=%2d %-3s  mean=%.3e std=%.3e best=%.3e worst=%.3e\n', ...
                Pr.name, D, opt{o,1}, mean(e), std(e), min(e), max(e));
        end

        % convergence figure
        f = figure('Theme','light','Position',[100 100 760 480]);
        col = lines(nOpt);
        for o = 1:nOpt
            m = mean(max(curves{o}, 1e-12), 1);
            semilogy(cp, m, 'LineWidth', 1.8, 'Color', col(o,:)); hold on;
        end
        xlabel('function evaluations');
        ylabel('error (f - f^*), log scale');
        legend(opt(:,1), 'Location','northeast'); grid on;
        title(sprintf('%s, D = %d: convergence (mean of %d runs)', ...
            Pr.name, D, nRun));
        exportgraphics(f, fullfile(figdir, ...
            sprintf('p3_conv_%s_D%d.png', funcs{fi}, D)), 'Resolution',160);
        close(f);
    end
end
fprintf('Total compute time %.1f s\n', toc(t0));

% ---- results table ----
T = cell2table(rows, 'VariableNames', ...
    {'Function','D','Optimiser','MeanErr','StdErr','BestErr','WorstErr'});
writetable(T, fullfile(resdir,'p3_results.csv'));

fid = fopen(fullfile(resdir,'p3_results.txt'),'w');
fprintf(fid,['CEC2005 comparison: final error (f - f*) over %d runs, ' ...
    'budget 10000*D FES\n\n'], nRun);
fprintf(fid,'%-22s %3s %-4s %12s %12s %12s %12s\n', ...
    'Function','D','Opt','Mean','Std','Best','Worst');
for i = 1:size(rows,1)
    fprintf(fid,'%-22s %3d %-4s %12.4e %12.4e %12.4e %12.4e\n', rows{i,:});
end
fclose(fid);

fid = fopen(fullfile(resdir,'p3_results.txt'),'r'); t = fread(fid,'*char')'; fclose(fid);
fprintf('=== p3_results.txt ===\n%s\n', t);
fprintf('Part 3 done.\n');
end
