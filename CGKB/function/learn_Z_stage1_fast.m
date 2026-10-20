function Z = learn_Z_stage1_fast(X, gamma1, gamma2, opts)
% learn_Z_stage1_fast  Stage-1 graph learning for MVCGKB (fast implementation).
% Solves for the per-view self-expressive matrices Z_v; the optimization loop
% uses the following speed-ups over a naive implementation:
%
%   1) Init          : rank + P obtained from a single SVD (original did two).
%   2) S sub-problem : when the latent dim mv < n, solve the n x n system
%                      (mu*I + 2*gamma*H'*H) * S = Right  via the Woodbury
%                      formula, reducing it to an mv x mv solve.
%   3) prox_tnn      : real tensors have conjugate-symmetric FFT slices, so
%                      only the first half of the frontal slices needs an SVD;
%                      each SVD is computed via the Hermitian Gram matrix
%                      (eig) instead of a full complex zgesdd.  Same result,
%                      ~2-4x faster.
%
% Solve:  sum_v ||X_v - P_v H_v||_F^2 + gamma sum_v ||H_v - H_v S_v||_F^2
%         + eta * ||S||_TNN,   s.t. P_v' P_v = I

if nargin < 4
    opts = struct();
end

if ~isfield(opts, 'max_iter'), opts.max_iter = 200; end
if ~isfield(opts, 'tol'),      opts.tol      = 1e-6; end
if ~isfield(opts, 'mu'),       opts.mu       = 1e-1; end
if ~isfield(opts, 'rho'),      opts.rho      = 1.2;  end
if ~isfield(opts, 'mu_max'),   opts.mu_max   = 1e6;  end

max_iter = opts.max_iter;
tol      = opts.tol;
mu       = opts.mu;
rho      = opts.rho;
mu_max   = opts.mu_max;

%% Dimensions
V = length(X);
n = size(X{1}, 2);
for v = 2:V
    if size(X{v}, 2) ~= n
        error('All views must have the same number of samples.');
    end
end

%% Initialization (single SVD per view: rank + P)
P = cell(V, 1);
H = cell(V, 1);
Z = zeros(n, n, V);
J = zeros(n, n, V);
Y = zeros(n, n, V);
mv_list = zeros(V, 1);

for v = 1:V
    Xv = X{v};
    if issparse(Xv)
        Xv = full(Xv);   % svd does not support sparse; densify (n is small)
    end
    [U, Ss, ~] = svd(Xv, 'econ');
    mv = sum(diag(Ss) > max(size(Xv)) * eps(Ss(1,1)));   % same tolerance as rank()
    mv_list(v) = mv;
    P{v} = U(:, 1:mv);
    H{v} = P{v}' * Xv;
    Z(:,:,v) = constructW_PKN(Xv, 15);
end

%% Main ADMM iteration
for iter = 1:max_iter

    Z_old = Z;

    % Step 1: Update P  (polar factor of M = X_v * H_v')
    % Tall views (dim_v >> mv): compute the polar factor through the Gram
    % matrix (M'*M, eig) which avoids the expensive full SVD; a tall M'*M
    % GEMM is much cheaper than zgesdd.  Rank-deficient case falls back to
    % the exact SVD, so P always equals U*V' up to round-off.
    for v = 1:V
        M = X{v} * H{v}';
        if size(M, 1) > 2 * size(M, 2)          % clearly tall -> Gram path
            G = M' * M;                          % mv x mv Hermitian
            G = (G + G') / 2;
            [Ve, De] = eig(G);
            d = real(diag(De));
            thr = max(d) * 1e-12;
            keep = d > thr;
            if all(keep)
                s = sqrt(max(d, 0));
                P{v} = M * (Ve * diag(1 ./ s) * Ve');   % = U*V'
            elseif ~any(keep)
                P{v} = M;                              % M ~ 0
            else
                [U, ~, Vv] = svd(M, 'econ');           % exact fallback
                P{v} = U * Vv';
            end
        else
            [U, ~, Vv] = svd(M, 'econ');
            P{v} = U * Vv';
        end
    end

    % Step 2: Update H
    for v = 1:V
        Xv = X{v}; Pv = P{v};
        Q = eye(n) - Z(:,:,v);
        M = eye(n) + gamma1 * (Q * Q');          % SPD -> Cholesky solve
        H{v} = (Pv' * Xv) / M;
    end

 % Step 3: Update Z
    for v = 1:V
        Hv = H{v};
        G  = Hv' * Hv;
        Right = 2 * gamma1 * G + mu * (J(:,:,v) - Y(:,:,v));
        mv = mv_list(v);
        if mv > 0 && mv < n && gamma1 ~= 0
            invmu = 1 / (mu+2);
            K = (Hv * Hv') * invmu + eye(mv) / (2 * gamma1);   % mv x mv SPD
            T = K \ (Hv * Right);                              % mv x n
            Z(:,:,v) = invmu * (Right - invmu * (Hv' * T));
        else
            Z(:,:,v) = (2 * gamma1 * G + (mu+2) * eye(n)) \ Right; % direct SPD solve
        end
    end
    % Step 4: Update J via fast TNN proximal operator
    J = prox_tnn_fast(Z + Y, gamma2 / mu);

    % Step 5: Update dual variable
    Y = Y + Z - J;

    % Step 6: Convergence check
    primal_res = norm(Z(:) - J(:)) / max(1, norm(Z(:)));
    relative_change = norm(Z(:) - Z_old(:)) / max(1, norm(Z_old(:)));
    if primal_res < tol && relative_change < tol
        break;
    end

    % Step 7: Update penalty parameter
    mu = min(rho * mu, mu_max);
end

end


% ==============================================================
% Fast tensor nuclear norm proximal operator.
% Y is real along mode 3, so Yf(:,:,k) = conj(Yf(:,:,n3+2-k)).
% Soft-thresholding commutes with conjugation => SVD only the first
% half of the frequency slices and mirror the rest.
% ==============================================================
function X = prox_tnn_fast(Y, tau)

[n1, n2, n3] = size(Y);

Yf = fft(Y, [], 3);
Xf = zeros(n1, n2, n3);

nh = floor(n3 / 2) + 1;          % number of independent slices

for k = 1:nh
    M = Yf(:,:,k);
    G = M' * M;                  % Hermitian n2 x n2
    G = (G + G') / 2;            % enforce exact Hermitian symmetry
    [V, D] = eig(G);
    d = real(diag(D));
    thr = max(d) * 1e-12;        % rank tolerance (matches svd precision)
    keep = d > thr;
    if ~any(keep)
        Xf(:,:,k) = 0;           % all-zero slice
        continue;
    end
    s = sqrt(d(keep));
    U = M * V(:, keep) * diag(1 ./ s);   % left singular vectors
    sigma = max(s - tau, 0);             % soft-threshold singular values
    Xf(:,:,k) = U * diag(sigma) * V(:, keep)';
end

% mirror the remaining slices via conjugate symmetry
for k = nh+1:n3
    Xf(:,:,k) = conj(Xf(:,:,n3+2-k));
end

X = real(ifft(Xf, [], 3));

end
