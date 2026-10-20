function S = sinkhorn(S)
    for k = 1:100
        S = S ./ (sum(S, 2) + eps); % 行归一化 (S * 1 = 1)
        S = S ./ (sum(S, 1) + eps); % 列归一化 (S' * 1 = 1)
        row_errors = abs(sum(S, 2) - 1);
        col_errors = abs(sum(S, 1) - 1);
        max_error = max([row_errors; col_errors']);
        % 检查收敛
        if max_error < 1e-5
            break;
        end
    end
end