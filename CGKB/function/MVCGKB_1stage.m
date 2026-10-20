function [result,A] = MVCGKB_1stage(X, gamma1, gamma2, Y, opts)
% MVCGKB_1stage  MVCGKB 的第一阶段 (只有共识图学习), 通过谱聚类得到聚类结果.
%
% 与 MVCGKB 的第一阶段数学等价, 不做 k-block 学习:
%   - learn_Z_stage1_fast : 快速 TNN prox (半谱 + Gram-eig), Woodbury 求解 Z,
%                           单次 SVD 初始化
%   - 共识图 Zc 的构建改为三维向量化
%
% 输出:
%   result : Clustering8Measure 指标 [ACC, ARI, NMI, ...]
%   S      : sinkhorn 双随机化后的共识图 (n x n)

if nargin < 5 || isempty(opts)
    opts = struct();
end

% ---- option defaults ----
if ~isfield(opts, 'max_iter'), opts.max_iter = 500;  end
if ~isfield(opts, 'tol'),      opts.tol      = 1e-3; end
if ~isfield(opts, 'mu'),       opts.mu       = 1e-1; end
if ~isfield(opts, 'rho'),      opts.rho      = 1.5;  end
if ~isfield(opts, 'mu_max'),   opts.mu_max   = 1e8;  end
if ~isfield(opts, 'verbose'),  opts.verbose  = true; end

%% parameter set
n = size(X{1}, 2);
k = length(unique(Y));

%% graph learning (fast stage-1, same algorithm)
Z = learn_Z_stage1_fast(X, gamma1, gamma2, opts);   % 不加噪声

% consensus graph (vectorized build, same as the per-view loop)
Zc = sum(abs((Z + permute(Z, [2 1 3])) / 2), 3);
A  = sinkhorn(Zc);
label = SpectralClustering(Zc, k);

result = Clustering8Measure(Y, label);

end
