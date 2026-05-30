function cec_landscapes()
%CEC_LANDSCAPES  Surface/contour plots of F1 and F9 at D = 2 for the report.
here   = fileparts(mfilename('fullpath'));
figdir = fullfile(here,'..','figures');
if ~exist(figdir,'dir'); mkdir(figdir); end
set(groot,'defaultFigureColor','w');

specs = {'F1', 60; 'F9', 4};     % name, plot half-range around shift
for s = 1:size(specs,1)
    P = cec_problem(specs{s,1}, 2);
    rad = specs{s,2};
    c = P.o;                                  % centre on the shifted optimum
    g = linspace(-rad, rad, 120);
    [Xg, Yg] = meshgrid(c(1)+g, c(2)+g);
    Z = zeros(size(Xg));
    for i = 1:numel(Xg)
        Z(i) = P.f([Xg(i) Yg(i)]);
    end
    f = figure('Theme','light','Position',[100 100 820 360]);
    subplot(1,2,1);
    surf(Xg, Yg, Z, 'EdgeColor','none'); view(135,35);
    xlabel('x_1'); ylabel('x_2'); zlabel('f'); title([P.name ' surface']);
    subplot(1,2,2);
    contourf(Xg, Yg, Z, 25, 'LineColor','none'); hold on;
    plot(c(1), c(2), 'rp', 'MarkerSize',12, 'MarkerFaceColor','r');
    xlabel('x_1'); ylabel('x_2'); title([P.name ' contour (star = optimum)']);
    axis tight;
    exportgraphics(f, fullfile(figdir, ...
        sprintf('p3_landscape_%s.png', specs{s,1})), 'Resolution',160);
    close(f);
end
fprintf('CEC landscapes written.\n');
end
