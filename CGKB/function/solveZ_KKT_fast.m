function Z = solveZ_KKT_fast(A, F, lambda1, lambda2, verbose)
% solveZ_KKT_fast  k-block 子问题的半光滑牛顿法 (Schur 补加速版).
%
% 朴素每个牛顿步:  d = pinv(J) * res          (对 2n x 2n 的 J 做 SVD 伪逆)
% 本版:           用 Schur 补把 J d = res 化为 n x n 的 Sc v = rhs, 用
%                 lsqminnorm 求解 v, 再回代 u.
%
% J = [-diag(D_row), -P; -P', -diag(D_col)], 零空间为 span{[1; -1]}.
% 由于 Z 只依赖于 alpha*1' + 1*beta', 任意满足 J d = res 的解(差一个零空间
% 向量)对应的 Z 更新完全相同, 因此结果与 pinv 版本一致.
%
% 原始问题:
%   min ||A-Z||_F^2 + lambda1*Tr(F'*L_Z*F) + lambda2*||Z||_F^2
%   s.t. Z>=0, Z*1=1, Z'*1=1
% 等价于 min ||Z-B||_F^2,  B = (A + (lambda1/2)*F*F')/(1+lambda2)
% 对偶形式 Z = max(0, B - alpha*1' - 1*beta'), 半光滑牛顿 + 回溯线搜索.

if nargin < 5, verbose = false; end

    n = size(A, 1);

    M = F * F';
    B = (A + (lambda1/2) * M) / (1 + lambda2);

    alpha = zeros(n, 1);
    beta  = zeros(n, 1);

    max_iter = 100;
    tol      = 1e-6;

    for iter = 1:max_iter
        C = B - alpha * ones(1,n) - ones(n,1) * beta';
        Z = max(0, C);

        r_row = sum(Z, 2) - 1;   % n x 1
        r_col = sum(Z, 1)' - 1;  % n x 1
        res   = [r_row; r_col];
        res_norm = norm(res);

        if verbose, fprintf('Iter %3d: residual = %.6e\n', iter, res_norm); end
        if res_norm < tol
            break;
        end

        % ---- Newton direction: J d = res, J = [-D_row, -P; -P', -D_col] ----
        active = (C > 0);
        D_row = sum(active, 2);   % row active counts
        D_col = sum(active, 1)';  % col active counts

        if all(D_row > 0)
            P = double(active);
            invDr = 1 ./ D_row;
            % Schur complement of J w.r.t. top-left block:
            %   Sc = P' * diag(1./D_row) * P - diag(D_col)
            %   rhs = r_col - P' * (r_row ./ D_row)
            Sc = (P' * (P .* invDr)) - diag(D_col);
            rhs = r_col - (P' * (r_row .* invDr));
            v = lsqminnorm(Sc, rhs);          % min-norm solve (Sc singular)
            u = -(r_row + P * v) .* invDr;    % back-substitute
            d = [u; v];
        else
            % degenerate row (empty active set) - fall back to full system
            J = [-diag(D_row), -double(active); -double(active)', -diag(D_col)];
            d = lsqminnorm(J, res);
        end

        % ---- backtracking line search ----
        step = 1.0;
        for ls = 1:20
            alpha_new = alpha - step * d(1:n);
            beta_new  = beta  - step * d(n+1:2*n);

            C_new = B - alpha_new * ones(1,n) - ones(n,1) * beta_new';
            Z_new = max(0, C_new);
            r_new = [sum(Z_new,2)-1; sum(Z_new,1)'-1];

            if norm(r_new) < res_norm
                alpha = alpha_new;
                beta  = beta_new;
                break;
            end
            step = step * 0.5;
        end
    end

    Z = max(0, B - alpha * ones(1,n) - ones(n,1) * beta');
    Z(Z < 1e-12) = 0;   % numerical cleanup

    if verbose, fprintf('>>> Final residual: %.6e\n', norm([sum(Z,2)-1; sum(Z,1)'-1])); end
end
