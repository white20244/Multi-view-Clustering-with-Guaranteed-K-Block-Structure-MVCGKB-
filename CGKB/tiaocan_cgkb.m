%% MVCGKB 两阶段调参
%   第一阶段: 用 MVCGKB_1stage 网格搜索 gamma1/gamma2 的最优值
%   第二阶段: 将第一阶段最优的 gamma1/gamma2 输入 MVCGKB, 固定不变, 网格搜索 lambda1/lambda2
%   最终得到四个参数的最优值 [gamma1, gamma2, lambda1, lambda2]
%%
clear;
clc;
tic
addpath(genpath(fileparts(mfilename('fullpath'))));
% ====================== 【保存路径设置】======================
load("dataset\yale_3fea.mat");
dataName='prokaryotic_res';
savePath = './res/';
if ~exist(savePath,'dir')
    mkdir(savePath);
end

filename = fullfile(savePath, [dataName '.mat']);
% ==========================================================

% 参数设置
cluster_num = length(unique(Y)); % 类数
viewnum = length(X); % 视图数
N = length(Y); % 样本数目
k = cluster_num;

% 指标列索引(在 result 中): ACC=1, ARI=2, NMI=3
% 以该指标作为"最优"的评判准则(默认 ACC, 可改为 3 用 NMI)
sel_idx = 1;

% 阈值
opts = struct();
opts.verbose = false;

% 数据转置 + 标准化(每个样本 L2 归一化)
for i=1:viewnum
    X{i}=normalize(X{i},2,'range',[0,1])';
end

%% ====================== 参数网格 ======================
% 第一阶段(MVCGKB_1stage): gamma1, gamma2
% 参数尺度依据(针对 NormalizeFea 的 L2 归一化, 列范数 O(1)):
%   gamma1 : 自表达项 ||H-HZ|| 权重, 参考 0.4~1, 留余量到 5
%   gamma2 : 张量核范数权重, 软阈值 tau=gamma2/mu, mu 初始 1e-3, 故取较小值
gamma1_list=[0.01,0.1,0.5,1,5];
gamma2_list=[0.1,1,5,10,15,20];

% 第二阶段(MVCGKB): lambda1, lambda2
% 依据 k-block 子问题: B = (A + (lambda1/2)*F*F')/(1+lambda2)
%   A 为 sinkhorn 后的双随机矩阵(元素量级 ~1/n), F*F' 元素量级 ~k/n
%   lambda1 控制 k-block 结构强度, lambda2 控制 Z 的平滑(越大 Z 越软)
%   demo 中参考值 lambda1=5, lambda2=10
lambda1_list=[0.01,0.1,1,5,10];
lambda2_list=[0.01,0.1,1,5,10];

%% ====================== 第一阶段: 调 gamma1/gamma2 ======================
fprintf('========== 第一阶段: MVCGKB_1stage 调 gamma1/gamma2 ==========\n');
N1 = length(gamma2_list)*length(gamma1_list);
Result1 = cell(1, N1);   % 每行 [gamma1, gamma2, ACC, ARI, NMI]
score1  = zeros(1, N1);  % 用于挑选最优的指标值
Num1 = 0;

for i=1:length(gamma2_list)
    for kk=1:length(gamma1_list)
        Num1=Num1+1;%记录轮数
        gamma2=gamma2_list(i);
        gamma1=gamma1_list(kk);
        [result,~] = MVCGKB_1stage(X,gamma1,gamma2,Y,opts);

        Result1{Num1} = [gamma1,gamma2,result(1),result(2),result(3)];
        score1(Num1)  = result(sel_idx);

        % ====================== 【核心：每轮实时保存】======================
        save(filename, 'Result1','score1','Num1','-v7.3');
        % =================================================================
    end
end

% 输出第一阶段最优结果
[~,best_idx1] = max(score1);
best1 = Result1{best_idx1};
gamma1_opt = best1(1);
gamma2_opt = best1(2);

kk_top=3;%输出最大 k 个指标对应的参数
[~,index1]=maxk(score1,kk_top);
top1 = zeros(kk_top, size(Result1{1},2));
for i=1:kk_top
    top1(i,:)=Result1{index1(i)};
end
disp('第一阶段最优参数 [gamma1, gamma2, ACC, ARI, NMI]:');
disp(top1);
fprintf('>>> 选定并固定: gamma1=%.4g, gamma2=%.4g\n\n', gamma1_opt, gamma2_opt); 

%% ====================== 第二阶段: 调 lambda1/lambda2 ======================
fprintf('========== 第二阶段: MVCGKB 调 lambda1/lambda2 ==========\n');
N2 = length(lambda1_list)*length(lambda2_list);
Result2 = cell(1, N2);   % 每行 [gamma1, gamma2, lambda1, lambda2, ACC, ARI, NMI, flag]
score2  = zeros(1, N2);
Num2 = 0;

for i=1:length(lambda1_list)
    for j=1:length(lambda2_list)
        Num2=Num2+1;%记录轮数
        lambda1=lambda1_list(i);
        lambda2=lambda2_list(j);
        [result, ~, flag] = MVCGKB(X,gamma1_opt,gamma2_opt,lambda1,lambda2,Y,opts);
        Result2{Num2} = [gamma1_opt,gamma2_opt,lambda1,lambda2, ...
                         result(1),result(2),result(3),flag];
        score2(Num2)  = result(sel_idx);

        % ====================== 【核心：每轮实时保存】======================
        save(filename, 'Result2','score2','Num2','-v7.3');
        % =================================================================
    end
end

% 输出第二阶段最优结果
[~,best_idx2] = max(score2);
best2 = Result2{best_idx2};
lambda1_opt = best2(3);
lambda2_opt = best2(4);

[~,index2]=maxk(score2,kk_top);
top2 = zeros(kk_top, size(Result2{1},2));
for i=1:kk_top
    top2(i,:)=Result2{index2(i)};
end
disp('第二阶段最优参数 [gamma1, gamma2, lambda1, lambda2, ACC, ARI, NMI, flag]:');
disp(top2);

%% ====================== 最终结果 ======================
final_opt = [gamma1_opt, gamma2_opt, lambda1_opt, lambda2_opt, ...
             best2(5), best2(6), best2(7)];
fprintf('\n========== 最终四个参数最优值 ==========\n');
fprintf('gamma1 = %.4g\n', gamma1_opt);
fprintf('gamma2 = %.4g\n', gamma2_opt);
fprintf('lambda1 = %.4g\n', lambda1_opt);
fprintf('lambda2 = %.4g\n', lambda2_opt);
fprintf('对应指标: ACC=%.4f, ARI=%.4f, NMI=%.4f (flag=%d)\n', ...
    best2(5), best2(6), best2(7), best2(8));

% 最终保存
save(filename, 'Result1','score1','Result2','score2','final_opt','-v7.3');
toc