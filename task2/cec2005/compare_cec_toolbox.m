function compare_cec_toolbox()
%COMPARE_CEC_TOOLBOX  Cross-check our optimisers against the Global
%   Optimization Toolbox (ga, particleswarm, simulannealbnd) on the same
%   two CEC'2005 functions, same FES budget, 15 runs. Produces a side table.
%
%   STW7085CEM Task 2, Part 3 (validation).

here   = fileparts(mfilename('fullpath'));
resdir = fullfile(here,'..','results');
if ~exist(resdir,'dir'); mkdir(resdir); end
if ~(license('test','GADS_Toolbox') && ~isempty(which('particleswarm')))
    fprintf('Global Optimization Toolbox not available; skipping.\n'); return;
end
warning('off','all');

funcs = {'F1','F9'};
Ds    = [2 10];
nRun  = 15;
rows  = cell(0,7);
t0 = tic;
for fi = 1:numel(funcs)
    for di = 1:numel(Ds)
        D = Ds(di);
        Pr = cec_problem(funcs{fi}, D);
        maxFES = 10000*D;
        lb = Pr.lb; ub = Pr.ub; fopt = Pr.fopt;
        obj = @(x) Pr.f(x);                    % accepts a 1-by-D row

        eGA = zeros(nRun,1); ePSO = zeros(nRun,1); eSA = zeros(nRun,1);
        for r = 1:nRun
            rng(2000+r);
            o1 = optimoptions('ga','PopulationSize',50, ...
                'MaxGenerations',round(maxFES/50),'Display','off');
            [~,v] = ga(obj, D, [],[],[],[], lb, ub, [], o1);
            eGA(r) = v - fopt;

            o2 = optimoptions('particleswarm','SwarmSize',40, ...
                'MaxIterations',round(maxFES/40),'Display','off');
            [~,v] = particleswarm(obj, D, lb, ub, o2);
            ePSO(r) = v - fopt;

            o3 = optimoptions('simulannealbnd', ...
                'MaxFunctionEvaluations',maxFES,'Display','off');
            x0 = lb + (ub-lb).*rand(1,D);
            [~,v] = simulannealbnd(obj, x0, lb, ub, o3);
            eSA(r) = v - fopt;
        end
        rows(end+1,:) = {Pr.name, D, 'ga()',            mean(eGA), std(eGA), min(eGA), max(eGA)}; %#ok<AGROW>
        rows(end+1,:) = {Pr.name, D, 'particleswarm()', mean(ePSO),std(ePSO),min(ePSO),max(ePSO)}; %#ok<AGROW>
        rows(end+1,:) = {Pr.name, D, 'simulannealbnd()',mean(eSA), std(eSA), min(eSA), max(eSA)}; %#ok<AGROW>
        fprintf('%s D=%d done\n', Pr.name, D);
    end
end
fprintf('Toolbox cross-check time %.1f s\n', toc(t0));

T = cell2table(rows,'VariableNames', ...
    {'Function','D','Optimiser','MeanErr','StdErr','BestErr','WorstErr'});
writetable(T, fullfile(resdir,'p3_toolbox.csv'));

fid = fopen(fullfile(resdir,'p3_toolbox.txt'),'w');
fprintf(fid,'CEC2005 toolbox cross-check: error (f - f*) over %d runs\n\n', nRun);
fprintf(fid,'%-22s %3s %-18s %12s %12s %12s %12s\n', ...
    'Function','D','Optimiser','Mean','Std','Best','Worst');
for i = 1:size(rows,1)
    fprintf(fid,'%-22s %3d %-18s %12.4e %12.4e %12.4e %12.4e\n', rows{i,:});
end
fclose(fid);
fid = fopen(fullfile(resdir,'p3_toolbox.txt'),'r'); t = fread(fid,'*char')'; fclose(fid);
fprintf('=== p3_toolbox.txt ===\n%s\n', t);
end
