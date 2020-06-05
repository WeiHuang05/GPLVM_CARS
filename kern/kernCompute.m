function K = kernCompute(model, i)

% KERNCOMPUTE Compute the kernel given the parameters and X.


x = model.X_u;
ell(1) = (model.kern.inverseWidth(i))^0.5;            % characteristic length scale
ell(2) = (model.kern.inverseWidth(i+model.d))^0.5;
ell(3) = (model.kern.inverseWidth(i+2*model.d))^0.5;
sf2 = model.kern.variance(i);                      % signal variance
K = sq_dist(x'.*ell);                               % symmetric matrix Kxx
K = sf2*exp(-K/2);

