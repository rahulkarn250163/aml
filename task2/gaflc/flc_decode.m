function fis = flc_decode(x, tmpl)
%FLC_DECODE  Rebuild a mamfis from a chromosome (with repair).
%   Free breakpoints are written from x in enumeration order; each membership
%   function's breakpoints are then clamped to its variable range and sorted
%   so the resulting fuzzy sets are always valid.
fis = tmpl.base;
gi  = 1;
for g = 1:numel(tmpl.groups)
    G = tmpl.groups(g);
    p = G.base;
    for f = G.freeIdx
        p(f) = x(gi);
        gi   = gi + 1;
    end
    p = min(max(p, G.lo), G.hi);   % clamp to universe of discourse
    p = sort(p);                   % enforce a <= b <= c (<= d)
    if strcmp(G.kind,'in')
        fis.Inputs(G.v).MembershipFunctions(G.m).Parameters = p;
    else
        fis.Outputs(G.v).MembershipFunctions(G.m).Parameters = p;
    end
end
end
