function gKern = kernStochasticGradient(userId, model, kern, x, gK_uu)

if model.qinverse
    gKernlengcs = zeros(1,model.q);
else
    gKernlengcs = zeros(1,model.d);
end


[k, dist2xx] = rbfard2KernCompute(model, userId);
covGradK = gK_uu.*k;
    
if model.onevariance
   gKernvar =  sum(sum(gK_uu.*k))/kern.variance;
else
   gKernvar =  zeros(1, model.d); 
   gKernvar(userId) =  sum(sum(gK_uu.*k))/kern.variance(userId);
end
   
for q = 1:size(x, 2)
     gleng(q)  =  -(sum(covGradK*(x(:, q).*x(:, q))) -x(:, q)'*covGradK*x(:, q));
end
    
if model.qinverse
    gKernlengcs = gKernlengcs + gleng; 
else
    gKernlengcs(userId) =  sum(gleng);
end
       

gKern = [gKernvar gKernlengcs];

 
if model.onevariance
    gKern(1) =  gKern(1)*expTransform(model.kern.variance, 'gradfact');
    start = 1;
else
    gKern(1:model.d) =  gKern(1:model.d).*expTransform(model.kern.variance, 'gradfact');
    start = model.d;
end

gKern(start+1:end) =  gKern(start+1:end).*expTransform(model.kern.inverseWidth, 'gradfact');
