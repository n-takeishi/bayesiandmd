function x = cnormrnd(mu, Gamma, n)
%CNORMRND Pseudorandom numbers from complex normal distribution.
%------------------------------------------------------------------------------

if nargin<3, n=1; end

d = size(mu,2);
mu_ = [real(mu),imag(mu)];
Sigma_ = 0.5*[real(Gamma),-imag(Gamma);imag(Gamma),real(Gamma)];
Sigma_ = 0.5*(Sigma_+Sigma_');
x_ = mvnrnd(mu_, Sigma_, n);
x = x_(:,1:d) + 1i*x_(:,d+1:end);

end