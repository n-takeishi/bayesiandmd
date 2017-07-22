function [sample, info] = bdmd(X, Y, K, option)
%BDMD Gibbs sampler for Bayesian DMD
%
%   [SAMPLE, INFO] = BDMD(X, Y, K, OPTION)
%
%   X and Y are <N x D> matrices of snapshots, where N is the number of
%   snapshots. K is the hyperparameter that specifies the number of dynamic
%   modes. The output struct, SAMPLE, is composed as follows:
%       SAMPLE.Lambda : Eigenvalues. <1 x K>
%       SAMPLE.W      : Dynamic modes. <K x D>
%       SAMPLE.Phi    : Values of eigenfunctions. <N x K>
%       SAMPLE.Sigma2 : Noise variance.
%   See defaultopt.m for the structure of OPTION.

%   See the following paper for more details:
%       Takeishi, Kawahara, Tabei & Yairi, "Bayesian Dynamic Mode
%       Decomposition," in Proc. of the 26th Int'l Joint Conf. on
%       Artificial Intelligence (IJCAI), 2017.
%------------------------------------------------------------------------------

addpath('./gigrnd/');

if nargin<4, option=[]; end
option = defaultopt(option);
[N, D] = size(X);

% hyperparameters
alpha = 1e-3;
beta = 1e-3;

% initialization
if ~option.dmdinit || K > D
    Lambda = 2*rand(1,K)-1 + 1i*(2*rand(1,K)-1);
    W = 2*rand(K,D)-1 + 1i*(2*rand(K,D)-1);
    Phi = 2*rand(N,K)-1 + 1i*(2*rand(N,K)-1);
else
    [Lambda, W, Phi] = sdmd(X.', Y.', K);
    Lambda = Lambda.';
    W = W.';
    Phi = Phi.';
    Lambda(isnan(Lambda)) = 1.0;
    W(isnan(W)) = 1.0;
    Phi(isnan(Phi)) = 1.0;
end

A = ones(1, K);
F1 = X - Phi*diag(A)*W;
F2 = Y - Phi*diag(Lambda)*diag(A)*W;
Sigma2 = sum(sum(F1.*conj(F1) + F2.*conj(F2)))/(2*N*D-1);
Nu2 = ones(1, K);
V2 = ones(K, D);
Tau2 = ones(1, K);

if option.spprior
    if isfield(option, 'initgamma')
        Gamma_v = option.initgamma*ones(1, K);
    else
        Gamma_v = D*sqrt(Sigma2)./sum(abs(W),2)';
    end
end
if option.ardprior
    if isfield(option, 'initgamma')
        Gamma_tau = option.initgamma;
    else
        Gamma_tau = K*sqrt(Sigma2)/sum(abs(A));
    end
end

% memory allocation
num_sample = floor((option.gibbsiter-option.burnin)/option.interval);
sample.Lambda = zeros(num_sample, K);
sample.W = zeros(num_sample, K, D);
sample.Phi = zeros(num_sample, N, K);
sample.Sigma2 = zeros(num_sample, 1);
sample.Nu2 = zeros(num_sample ,K);
sample.V2 = zeros(num_sample, K, D);
if option.ardprior
    sample.A = zeros(num_sample, K);
    sample.Tau2 = zeros(num_sample, K);
end

info = struct();
if option.calclikelihood
    info.likelihood = zeros(option.gibbsiter, 1);
end

if option.spprior
    info.Gamma_v = zeros(option.mcemiter, K);
end
if option.ardprior
    info.Gamma_tau = zeros(option.mcemiter, 1);
end

tic;

% main iteration of MC-EM
for mcem=1:option.mcemiter

    % E-step: Gibbs sampling
    count_sample = 1;
    for gibbs=1:option.gibbsiter
        
        % block update of Lambda
        Lambda_new = Lambda;
        for k=1:K
            Eta = get_Eta(Lambda, W, Phi, k, A);
            p = A(k)*A(k) * conj(W(k,:))*W(k,:).' * Phi(:,k)'*Phi(:,k) / Sigma2 ...
                + 1/Nu2(k);
            m = A(k) * conj(W(k,:)) ...
                * sum(bsxfun(@times,Y-Eta,conj(Phi(:,k))),1).' / p / Sigma2;
            Lambda_new(k) = cnormrnd(m, 1/p);
        end
        Lambda = Lambda_new;

        % block update of C
        W_new = W;
        for k=1:K
            if option.spprior
                tmp = Sigma2*V2(k,:);
            else
                tmp = V2(k,:);
            end
            Xi = get_Xi(W, Phi, k, A);
            Eta = get_Eta(Lambda, W, Phi, k, A);
            P = A(k)*A(k) * (1+conj(Lambda(k))*Lambda(k))/Sigma2 * Phi(:,k)'*Phi(:,k) * eye(D) ...
                + diag(1./tmp);
            P_inv = diag(1./diag(P));
            m = A(k) * (sum(bsxfun(@times,X-Xi,conj(Phi(:,k))),1) ...
                + conj(Lambda(k)) * sum(bsxfun(@times,Y-Eta,conj(Phi(:,k))),1))/Sigma2 * P_inv.';
            W_new(k,:) = cnormrnd(m, P_inv);
        end
        W = W_new;

        % block update of Phi
        P = diag(A)*conj(W)*W.'*diag(A) / Sigma2 ...
            + diag(conj(Lambda)) * diag(A) * conj(W)*W.' * diag(A) * diag(Lambda) / Sigma2 ...
            + eye(K);
        P = 0.5*(P+P'); % ensure Hermitian
        [UP,DP] = eig(P);
        P_inv = UP*diag(1./diag(DP))*UP';
        m = (X * W' * diag(A) ...
            + Y * W' * diag(A) * diag(conj(Lambda)))/Sigma2 * P_inv.';
        Phi = cnormrnd(m, P_inv, N);

        % update of Sigma2
        F1 = X - Phi*diag(A)*W;
        F2 = Y - Phi*diag(Lambda)*diag(A)*W;
        a = 2*N*D + alpha;
        b = sum(sum(conj(F1).*F1,2),1) + sum(sum(conj(F2).*F2,2),1) + beta;
        if option.spprior
            a = a + 0.5*K*D;
            b = b + 0.5*sum(sum(conj(W).*W./V2));
        end
        if option.ardprior
            a = a + 0.5*K;
            b = b + 0.5*sum((A.^2)./Tau2);
        end
        Sigma2 = 1/gamrnd(a, 1/b);

        % block update of Nu2
        %a = 1 + alpha;
        %b = beta + conj(Lambda).*Lambda;
        %Nu2 = 1./gamrnd(a, 1./b, 1, K);

        % block update of V2
        if option.spprior
            a = Gamma_v.^2;
            b = conj(W).*W/Sigma2;
            for k=1:K
                for d=1:D
                    V2(k,d) = gigrnd(0.5, a(k), b(k,d), 1);
                end
            end
        else
            a = 1 + alpha;
            b = conj(W).*W + beta;
            V2 = 1./gamrnd(a, 1./b, K, D);
        end

        % block update of A
        if option.ardprior
            A_new = A;
            for k=1:K
                Xi = get_Xi(W, Phi, k, A);
                Eta = get_Eta(Lambda, W, Phi, k, A);
                p = 2 * (conj(W(k,:))*W(k,:).' * Phi(:,k)'*Phi(:,k) * (1 + conj(Lambda(k))*Lambda(k)) ...
                    + 1/Tau2(k)) / Sigma2;
                p = real(p);
                m = 2 * real(conj(W(k,:)) * sum(bsxfun(@times,X-Xi,conj(Phi(:,k))),1).' ...
                    + conj(Lambda(k)) * conj(W(k,:)) * sum(bsxfun(@times,Y-Eta,conj(Phi(:,k))),1).') / Sigma2 / p;
                A_new(k) = normrnd(m, 1/p);
            end
            A = A_new;

            % block update of Tau2
            a = Gamma_tau^2;
            b = (A.^2)/Sigma2;
            for k=1:K
                Tau2(k) = gigrnd(0.5, a, b(k), 1);
            end
        end

        if gibbs > option.burnin && mod(gibbs-option.burnin, option.interval) == 0
            % save sample
            sample.Lambda(count_sample, :) = Lambda;
            sample.W(count_sample, :, :) = W;
            sample.Phi(count_sample, :, :) = Phi;
            sample.Sigma2(count_sample) = Sigma2;
            sample.Nu2(count_sample, :) = Nu2;
            sample.V2(count_sample, :, :) = V2;
            if option.ardprior
                sample.A(count_sample, :) = A;
                sample.Tau2(count_sample, :) = Tau2;
            end
            count_sample = count_sample + 1;
        end

        % calculate likelihood
        if option.calclikelihood
            F1 = X - Phi*diag(A)*W;
            F2 = Y - Phi*diag(Lambda)*diag(A)*W;
            info.likelihood(gibbs) = ...
                -N*D*log(Sigma2*pi) - sum(sum(conj(F1).*F1,2),1)/Sigma2 ...
                -N*D*log(Sigma2*pi) - sum(sum(conj(F2).*F2,2),1)/Sigma2;
        end

        % display
        if option.display && mod(gibbs,1000)==0
            fprintf('gibbs-iter=%d', gibbs);
            if gibbs <= option.burnin
                fprintf(' (burn-in)');
            end
            if option.calclikelihood
                fprintf('\t\t%e', info.likelihood(gibbs));
            end
            fprintf('\n');
        end
    end

    % M-step: empirical Bayes
    if option.spprior
        E_V2 = mean(sample.V2, 1);
        Gamma_v = sqrt(2*D./reshape(sum(E_V2,3), 1, K));
        info.Gamma_v(mcem,:) = Gamma_v;
        if option.display
            fprintf('--- MCEM update (%d): avg(Gamma_v)=%e ---\n', mcem, mean(Gamma_v));
        end
    end
    if option.ardprior
        E_tau2 = mean(sample.Tau2, 1);
        Gamma_tau = sqrt(2*K/sum(E_tau2));
        info.Gamma_tau(mcem) = Gamma_tau;
        if option.display
            fprintf('--- MCEM update (%d): Gamma_tau=%e ---\n', mcem, Gamma_tau);
        end
    end

end

info.duration = toc;

% impose artificial identifiablity constraint
if ~isempty(option.ictype) && ~strcmp(option.ictype, 'none')
    sample = sortsample(sample, option.ictype);
end

end

% ----------

function Xi = get_Xi(W, Phi, k, tau)
    K = size(W,1);
    idxk = [1:k-1,k+1:K];
    Xi = Phi(:,idxk)*diag(tau(idxk))*W(idxk,:);
end

function Eta = get_Eta(Lambda, W, Phi, k, tau)
    K = size(W,1);
    idxk = [1:k-1,k+1:K];
    Eta = Phi(:,idxk)*diag(Lambda(idxk))*diag(tau(idxk))*W(idxk,:);
end