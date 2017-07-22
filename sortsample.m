function sample = sortsample(sample, ictype)
%SORTSAMPLE Sort samples from Gibbs sampler to impose artificial
%identifiability constraint.
%------------------------------------------------------------------------------

num_sample = size(sample.Lambda,1);

% Types of artificial identifiability constraint:
if     strcmp(ictype, 'eigval_real')
    [~, sample.idx] = sort(real(sample.Lambda), 2, 'descend');
elseif strcmp(ictype, 'eigval_imag')
    [~, sample.idx] = sort(imag(sample.Lambda), 2, 'descend');
elseif strcmp(ictype, 'eigval')
    [~, sample.idx] = sort(sample.Lambda, 2, 'descend');
elseif strcmp(ictype, 'mode_abs')
    [~, sample.idx] = sort(sum(abs(sample.C),3), 2, 'descend');
elseif strcmp(ictype, 'mode_maxidx')
    [~, c_maxidx] = max(permute(sample.C,[2,3,1]), [], 2);
    c_maxidx = reshape(c_maxidx, size(c_maxidx,1), size(c_maxidx,3))';
    [~, sample.idx] = sort(c_maxidx, 2, 'ascend');
else
    error('no such sorting scheme.');
end

% Impose the selected constraint.
for i=1:num_sample
    idx = sample.idx(i,:);
    sample.Lambda(i,:) = sample.Lambda(i,idx);
    sample.W(i,:,:) = sample.W(i,idx,:);
    sample.Phi(i,:,:) = sample.Phi(i,:,idx);
    sample.Nu2(i,:) = sample.Nu2(i,idx);
    sample.V2(i,:,:) = sample.V2(i,idx,:);
    if isfield(sample, 'A')
        sample.A(i,:) = sample.A(i,idx);
    end
    if isfield(sample, 'Tau2')
        sample.Tau2(i,:) = sample.Tau2(i,idx);
    end
end

end