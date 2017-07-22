function [evalue, mode, efun, levec] = sdmd(X, Y, d_x)
%SDMD Exact DMD
%------------------------------------------------------------------------------

if nargin>2 && ~isempty(d_x)
    [U, S, V] = svd(X, 'econ');
    [diag_S, idx] = sort(diag(S), 'descend');
    U = U(:,idx(1:d_x)); S = diag(diag_S(1:d_x)); V = V(:,idx(1:d_x));
else
    [U, S, V] = svd(X);
    idx = abs(diag(S))>1e-4;
    U = U(:,idx); S = S(idx,idx); V = V(:,idx);
end
M = Y*V/S;
A_til = U'*M;
[w, D, z] = eig(A_til);

% normalization
N = conj(z'*w);
z = z/N;

[D,idx] = sort(diag(D),'descend');
phi = (M*w(:,idx))/diag(D); %phi = U*w(:,idx)
kap = U*z(:,idx);

% outputs
evalue = D;
mode = conj(kap);
efun = phi'*X;
levec = kap;
