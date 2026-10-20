%% parameter
%                 gamma1          gamma2          lambda1       lambda2    
%msrcv1           1               1               1             5
%yaleA            0.1             0.1             1             0.01
%yale             0.01            10              1             1
%ORL4             0.1             15              1             1
% 100leaves       0.01            1               10            5
%ORL3             0.01            10              5             10
% prokaryotic     0.5             10              1             0.1
% BBCSport        0.5             20              0.1           0.01
% Mnist           0.01            20              0.01          1
% BDGP            0.1             10              1             0.1
%% initialize
clear;
clc;
addpath(genpath(fileparts(mfilename('fullpath'))));
% load data
load("dataset\yale_3fea.mat");
%% 运行选项
opts = struct();
opts.max_iter = 500;
opts.tol = 1e-4;
opts.mu = 1e-1;
opts.rho = 1.5;
opts.mu_max = 1e8;
opts.verbose = false;
%% set
k= length(unique(Y)); % cluster number
viewnum = length(X); % view number
num_runs = 1; % run times
% normalize
for i=1:viewnum
    X{i}=normalize(X{i},2,'range',[0,1])';
end

%% parameter
gamma1=0.01;
gamma2=10;
lambda1=1;
lambda2=1;
%% result
acc_list = zeros(1, num_runs);
ari_list = zeros(1, num_runs);
nmi_list = zeros(1, num_runs);
tic
for run = 1:num_runs
[result, S, flag] = MVCGKB(X,gamma1,gamma2,lambda1,lambda2,Y,opts);
acc_list(run) = result(1);
ari_list(run) = result(2);
nmi_list(run) = result(3);

disp(run)
% disp(result)
end
toc
% average performance
disp('average performance:');
disp(['ACC: ', num2str(mean(acc_list)), ' ± ', num2str(std(acc_list))]);
disp(['ARI: ', num2str(mean(ari_list)), ' ± ', num2str(std(ari_list))]);
disp(['NMI: ', num2str(mean(nmi_list)), ' ± ', num2str(std(nmi_list))]);
