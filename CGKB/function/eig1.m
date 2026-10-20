function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end

if nargin < 3
    isMax = 1;
    isSym = 1;
end

if nargin < 4
    isSym = 1;
end

if isSym == 1
    A = max(A,A');
end
[v d] = eig(A);
d = real(diag(d));
if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');%��С��������
end

idx1 = idx(1:c);%ȡǰc��
eigval = d(idx1);%��Ӧ����ֵȡǰc��
eigvec = v(:,idx1);%��Ӧ��������ȡǰc��

eigval_full = d(idx);