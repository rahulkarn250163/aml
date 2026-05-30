function mse = flc_fitness(x, tmpl, X, y)
%FLC_FITNESS  Mean squared error of the decoded FLC over the dataset.
%   Lower is better. Invalid evaluations are penalised heavily so the GA
%   avoids them.
fis = flc_decode(x, tmpl);
try
    yhat = evalfis(fis, X);
catch
    mse = 1e6; return;
end
d   = yhat - y;
mse = mean(d.^2);
if ~isfinite(mse)
    mse = 1e6;
end
end
