function flc_evidence()
%FLC_EVIDENCE  Generate every Part 1 figure for the Task 2 report.
%   Writes PNGs to ../figures and a rule-activation / scenario log to
%   ../results. Runs headless under: matlab -batch "flc_evidence".

here   = fileparts(mfilename('fullpath'));
figdir = fullfile(here,'..','figures');
resdir = fullfile(here,'..','results');
if ~exist(figdir,'dir'); mkdir(figdir); end
if ~exist(resdir,'dir'); mkdir(resdir); end

fis = build_flc();
set(groot,'defaultFigureColor','w');

% ----------------------------------------------------------- 1. MF plots
labels = {'temperature','activity','hvac'};
kinds  = {'input','input','output'};
idx    = [1 2 1];
f = figure('Theme','light','Position',[100 100 900 700]);
for i = 1:3
    subplot(3,1,i);
    plotmf(fis,kinds{i},idx(i));
    title(sprintf('Membership functions: %s', labels{i}), ...
        'Interpreter','none');
    xlabel(labels{i}); ylabel('degree');
    grid on;
end
exportgraphics(f, fullfile(figdir,'p1_membership_functions.png'), ...
    'Resolution',160);
close(f);

% Individual MF plots too (clearer for the report if needed).
for i = 1:3
    f = figure('Theme','light','Position',[100 100 700 320]);
    plotmf(fis,kinds{i},idx(i));
    title(sprintf('%s membership functions', labels{i}),'Interpreter','none');
    xlabel(labels{i}); ylabel('degree'); grid on;
    exportgraphics(f, fullfile(figdir, ...
        sprintf('p1_mf_%s.png',labels{i})),'Resolution',160);
    close(f);
end

% --------------------------------------------------- 2. FIS block diagram
f = figure('Theme','light','Position',[100 100 700 420]);
plotfis(fis);
exportgraphics(f, fullfile(figdir,'p1_block_diagram.png'),'Resolution',160);
close(f);

% ------------------------------------------------------- 3. control surface
f = figure('Theme','light','Position',[100 100 720 560]);
gensurf(fis);
title('Control surface: hvac vs temperature and activity');
view(135,30);
exportgraphics(f, fullfile(figdir,'p1_control_surface.png'),'Resolution',160);
close(f);

% ----------------------------------------- 4. rule activation at a sample
% Operating point: warm afternoon, resident active.
pt = [29 7];
rulesFired = rule_activation(fis, pt);
f = figure('Theme','light','Position',[100 100 820 420]);
bar(rulesFired);
xlabel('rule number'); ylabel('firing strength');
title(sprintf(['Rule activation at temperature = %g, activity = %g' ...
    '   (hvac = %.1f)'], pt(1), pt(2), evalfis(fis,pt)));
grid on; ylim([0 1]);
exportgraphics(f, fullfile(figdir,'p1_rule_activation.png'),'Resolution',160);
close(f);

% --------------------------------------------------- 5. operational scenario
t = 0:0.25:24;                                  % hours across one day
temp = 20 + 8*exp(-((t-15).^2)/(2*3.0^2));      % warm afternoon peak ~28 C
act  = 1.5 + 6.5*exp(-((t-14).^2)/(2*4.0^2));   % active midday, calm at night
hv   = zeros(size(t));
for k = 1:numel(t)
    hv(k) = evalfis(fis,[temp(k) act(k)]);
end

f = figure('Theme','light','Position',[100 100 900 640]);
subplot(2,1,1);
yyaxis left;  plot(t,temp,'LineWidth',1.6); ylabel('temperature (C)');
yyaxis right; plot(t,act,'LineWidth',1.6);  ylabel('activity index');
xlabel('time of day (h)'); title('Operational scenario: inputs over one day');
grid on; xlim([0 24]);
subplot(2,1,2);
plot(t,hv,'LineWidth',1.8); hold on; yline(0,'k:');
xlabel('time of day (h)'); ylabel('hvac command');
title('Controller output (negative = cooling, positive = heating)');
grid on; xlim([0 24]); ylim([-100 100]);
exportgraphics(f, fullfile(figdir,'p1_operational_scenario.png'), ...
    'Resolution',160);
close(f);

% -------------------------------------------------------------- text log
fid = fopen(fullfile(resdir,'p1_evidence_log.txt'),'w');
fprintf(fid,'FLC: %s  (Mamdani, centroid)\n', fis.Name);
fprintf(fid,'Inputs: temperature[10 35], activity[0 10]; Output: hvac[-100 100]\n');
fprintf(fid,'Rules: %d\n\n', numel(fis.Rules));
fprintf(fid,'Rule activation at temperature=%g activity=%g:\n', pt(1),pt(2));
for r = 1:numel(rulesFired)
    if rulesFired(r) > 1e-6
        fprintf(fid,'  rule %2d firing strength %.3f\n', r, rulesFired(r));
    end
end
fprintf(fid,'  -> hvac output %.2f\n\n', evalfis(fis,pt));
fprintf(fid,'Scenario extremes:\n');
[mn,imn] = min(hv); [mx,imx] = max(hv);
fprintf(fid,'  max cooling %.1f at t=%.2fh (temp=%.1f act=%.1f)\n', ...
    mn,t(imn),temp(imn),act(imn));
fprintf(fid,'  max heating %.1f at t=%.2fh (temp=%.1f act=%.1f)\n', ...
    mx,t(imx),temp(imx),act(imx));
fclose(fid);
type(fullfile(resdir,'p1_evidence_log.txt'));

fprintf('\nPart 1 figures written to %s\n', figdir);
end


function fs = rule_activation(fis, pt)
%RULE_ACTIVATION  Firing strength of each rule at input point pt (AND=min).
nIn   = numel(fis.Inputs);
nRule = numel(fis.Rules);
% Pre-compute membership of each input value in every MF of its variable.
muIn = cell(1,nIn);
for v = 1:nIn
    mfs = fis.Inputs(v).MembershipFunctions;
    m   = zeros(1,numel(mfs));
    for j = 1:numel(mfs)
        m(j) = mfval(mfs(j), pt(v));
    end
    muIn{v} = m;
end
fs = zeros(nRule,1);
for r = 1:nRule
    ant = fis.Rules(r).Antecedent;   % MF index per input (0 = not used)
    vals = [];
    for v = 1:nIn
        if ant(v) > 0
            vals(end+1) = muIn{v}(ant(v)); %#ok<AGROW>
        end
    end
    if isempty(vals); fs(r) = 0; else; fs(r) = min(vals); end
    fs(r) = fs(r) * fis.Rules(r).Weight;
end
end
