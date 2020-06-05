function [gKern, gVarmeans, gVarcovars] = kernStochasticPsi0Gradient(userId, model, kern, vardist, covGrad)


if model.onevariance
    gKernvar = sum(covGrad*model.numData(userId));
else
    gKernvar = zeros(1,model.d);
    gKernvar(userId) = covGrad*model.numData(userId);  
end


if model.qinverse
   gKernlengcs = zeros(1,model.q);
else
   gKernlengcs = zeros(1,model.d);
end

gKern = [gKernvar gKernlengcs];

gVarmeans = zeros(vardist.nParams,1); 
gVarcovars = zeros(vardist.nParams,1);     

if model.onevariance
    gKern(1) =  gKern(1)*expTransform(kern.variance, 'gradfact');
    start = 1;
else
    gKern(1:model.d) =  gKern(1:model.d).*expTransform(kern.variance, 'gradfact');
    start = model.d;
end

gKern(start+1:end) =  gKern(start+1:end).*expTransform(kern.inverseWidth, 'gradfact');
 
end


