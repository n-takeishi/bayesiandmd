function option = defaultopt(option)
%DEFAULTOPT Default options of Bayesian DMD
%------------------------------------------------------------------------------

% Initialization by the standard DMD (if false, initialization by random values)
if ~isfield(option, 'dmdinit'),        option.dmdinit = true; end

% The total number of iterations of Gibbs sampling.
if ~isfield(option, 'gibbsiter'),      option.gibbsiter = 5000; end

% The number of iterations disposed for burn-in.
if ~isfield(option, 'burnin'),         option.burnin = 1000; end

% Intervals with which samples are adopted.
if ~isfield(option, 'interval'),       option.interval = 4; end

% Calculate likelihood at every iteration.
if ~isfield(option, 'calclikelihood'), option.calclikelihood = false; end

% Display outputs.
if ~isfield(option, 'display'),        option.display = true; end

% Type of artificial identifiability constraint. See sortsample.m for details.
if ~isfield(option, 'ictype'),         option.ictype = 'none'; end

% Set sparsity prior on dynamic modes. This is what is described in IJCAI paper.
if ~isfield(option, 'spprior'),        option.spprior = false; end

% Set ARD prior on dynamic modes. This is not described in IJCAI paper.
if ~isfield(option, 'ardprior'),       option.ardprior = false; end

% The number of iterations of MC-EM.
if ~isfield(option, 'mcemiter'),       option.mcemiter = 5; end
if ~option.ardprior && ~option.spprior,option.mcemiter = 1; end

end