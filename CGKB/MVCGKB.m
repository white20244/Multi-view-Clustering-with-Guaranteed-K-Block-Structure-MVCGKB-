function [result,S1, flag] = MVCGKB(X, gamma1, gamma2, lambda1,lambda2,Y, opts)
% MVCGKB  多视图聚类: 图学习 Stage1 + k-block 学习.
%
% 耗时环节均使用快速实现:
%   - learn_Z_stage1_fast : 快速 TNN prox (半谱 + Gram-eig), Woodbury 求解 Z,
%                           单次 SVD 初始化
%   - solveZ_KKT_fast     : 用 Schur 补 + lsqminnorm 代替 pinv(2n x 2n)
%   - 共识图 Zc 的构建改为三维向量化
%
% 聚类: 对 k-block 矩阵 S1 做连通分量.

if nargin < 7 || isempty(opts)
    opts = struct();
end

% ---- option defaults ----
if ~isfield(opts, 'max_iter'), opts.max_iter = 500;  end
if ~isfield(opts, 'tol'),      opts.tol      = 1e-3; end
if ~isfield(opts, 'mu'),       opts.mu       = 1e-1; end
if ~isfield(opts, 'rho'),      opts.rho      = 1.5;  end
if ~isfield(opts, 'mu_max'),   opts.mu_max   = 1e8;  end
if ~isfield(opts, 'verbose'),  opts.verbose  = true; end

max_iter = opts.max_iter;
tol = opts.tol;

%% parameter set
% viewnum = length(X);
n = size(X{1}, 2);
k = length(unique(Y));

%% graph learning (fast stage-1, same algorithm)
Z = learn_Z_stage1_fast(X, gamma1, gamma2, opts);   % 不加噪声

% consensus graph (vectorized build, same as the per-view loop)
Zc = sum(abs((Z + permute(Z, [2 1 3])) / 2), 3);
A = sinkhorn(Zc);

%% k-block learning
F = SPC(A,k);
flag = 0;
S = zeros(n,n);
for i = 1:max_iter
    S_old = S;
    S  = solveZ_KKT_fast(A, F, lambda1, lambda2, opts.verbose);
    S1 = (S + S') / 2;
    L  = eye(n) - S1;
    [F, ~, ev] = eig1(L, k, 0);

    thre = 1e-11;
    fn1  = sum(ev(1:k));   % sum of the k smallest eigenvalues of L
    if fn1 > thre
        lambda1 = lambda1 * 2;
    end
    %% convergence
    if norm(abs(S-S_old),'fro') < tol
        break
    end
end

%% clustering
% NOTE: cluster on the k-block matrix S1 (not the stage-1 consensus Zc,
% which is dense and yields a single connected component).
[label, binsize] = conncomp(graph(S1));
label = label';

if length(binsize) == k
    flag = 1;
else
    warning('MVCGKB:cluster', 'Cannot find %d clusters (found %d, lambda1 = %.3g).', ...
        k, length(binsize), lambda1);
end

result = Clustering8Measure(Y, label);

end
