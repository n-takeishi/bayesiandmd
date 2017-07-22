% Bayesian DMD on data generated from a simple linear system.

clear;
rng(1234, 'twister');
addpath('./bplot/');

%%

% generate data
N = 51;
lam_true = [0.9, 0.5];
data = zeros(N, 2);
for t=1:N, data(t,:) = lam_true.^(t-1); end
data = data + randn(size(data))*5e-2;

% Apply standard DMDs
[lam_sdmd] = sdmd(data(1:end-1,:).', data(2:end,:).', 2);

% Apply Bayesian DMD
option = struct(...
    'gibbsiter',      5000, ...
    'burnin',         1000, ...
    'interval',       4, ...
    'mcemiter',       1, ...
    'calclikelihood', true, ...
    'display',        true, ...
    'dmdinit',        false, ...
    'spprior',        false, ...
    'ictype',         'eigval_real');
[sample, info] = bdmd(data(1:end-1,:), data(2:end,:), 2, option);

%% plot results

figure;
hold on;
plot([1, 2], lam_true, 'bo', 'markerfacecolor', 'b', 'markersize', 8);
plot([1, 2], lam_sdmd, 'g^', 'markerfacecolor', 'g', 'markersize', 8);
bplot(real(sample.Lambda), 'linewidth', 1, 'width', 0.2, 'nomean', 'outliers');
hold off;
grid on;
xlabel('#dynamic mode');
ylabel('eigenvalue');
set(gca, 'xtick', [1 2]);
legend({'ground truth', 'standard DMD', 'Bayesian DMD (median)'});