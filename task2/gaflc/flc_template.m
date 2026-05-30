function tmpl = flc_template()
%FLC_TEMPLATE  Describe the FLC membership functions as a tunable chromosome.
%
%   The chromosome is a real-valued vector of membership-function breakpoints.
%   To keep the fuzzy partition valid we pin the two outermost anchors of the
%   first and last set on every variable (so each variable always spans its
%   full universe) and let the GA tune all remaining breakpoints. On decoding,
%   each membership function's breakpoints are clamped to the variable range
%   and sorted, which repairs any ordering the genetic operators break.
%
%   Returns tmpl with fields:
%     base   - the baseline mamfis (hand-designed FLC)
%     groups - per-membership-function metadata
%     x0     - baseline chromosome (the hand-designed breakpoints)
%     lb,ub  - per-gene lower/upper bounds
%     L      - chromosome length
%
%   STW7085CEM Task 2, Part 2.

fis = build_flc();
groups = struct('kind',{},'v',{},'m',{},'np',{},'freeIdx',{}, ...
                'lo',{},'hi',{},'base',{});

% --- inputs ---
for v = 1:numel(fis.Inputs)
    r   = fis.Inputs(v).Range;
    mfs = fis.Inputs(v).MembershipFunctions;
    M   = numel(mfs);
    for m = 1:M
        p  = mfs(m).Parameters; np = numel(p);
        groups(end+1) = mkgroup('in',v,m,np,pick_free(m,M,np), ...
            r(1),r(2),p); %#ok<AGROW>
    end
end

% --- output (single) ---
r   = fis.Outputs(1).Range;
mfs = fis.Outputs(1).MembershipFunctions;
M   = numel(mfs);
for m = 1:M
    p = mfs(m).Parameters; np = numel(p);
    groups(end+1) = mkgroup('out',1,m,np,pick_free(m,M,np), ...
        r(1),r(2),p); %#ok<AGROW>
end

% assemble chromosome in enumeration order
x0 = []; lb = []; ub = [];
for g = 1:numel(groups)
    G = groups(g);
    for f = G.freeIdx
        x0(end+1) = G.base(f); lb(end+1) = G.lo; ub(end+1) = G.hi; %#ok<AGROW>
    end
end

tmpl.base   = fis;
tmpl.groups = groups;
tmpl.x0     = x0(:);
tmpl.lb     = lb(:);
tmpl.ub     = ub(:);
tmpl.L      = numel(x0);
end

function f = pick_free(m, M, np)
%PICK_FREE  Indices of breakpoints the GA may tune for one MF.
if m == 1
    f = 3:np;          % pin the two leading anchors of the first set
elseif m == M
    f = 1:(np-2);      % pin the two trailing anchors of the last set
else
    f = 1:np;          % interior sets fully tunable
end
end

function s = mkgroup(kind,v,m,np,freeIdx,lo,hi,base)
s = struct('kind',kind,'v',v,'m',m,'np',np,'freeIdx',freeIdx, ...
           'lo',lo,'hi',hi,'base',base);
end
