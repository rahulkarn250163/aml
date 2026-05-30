function mu = mfval(mf, x)
%MFVAL  Membership degree of x in a fismf object (trimf/trapmf), API-robust.
%   Avoids relying on a specific evalmf signature across MATLAB versions.
p = mf.Parameters;
switch lower(mf.Type)
    case 'trimf'
        a = p(1); b = p(2); c = p(3);
        mu = zeros(size(x));
        if b > a, mu = max(mu, (x-a)/(b-a)); end
        if c > b, mu = min(mu, (c-x)/(c-b)); end
        mu(x==b) = 1;            % handle degenerate shoulders
        mu = max(0, min(1, mu));
    case 'trapmf'
        a = p(1); b = p(2); c = p(3); d = p(4);
        mu = zeros(size(x));
        left  = ones(size(x)); if b > a, left  = (x-a)/(b-a); end
        right = ones(size(x)); if d > c, right = (d-x)/(d-c); end
        mu = max(0, min(min(left, right), 1));
        mu(x>=b & x<=c) = 1;
    otherwise
        mu = evalmf(mf, x);     % fall back to built-in for other types
end
end
