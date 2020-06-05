function [gKern, gVarmeans, gVarcovars] = kernVardistPsi0Gradient(model, kern, vardist, covGrad)

[gKern, gVarmeans, gVarcovars] = rbfard2VardistPsi0Gradient(model, kern, vardist, covGrad);    

if model.onevariance
    gKern(1) =  gKern(1)*expTransform(model.kern.variance, 'gradfact');
    start = 1;
else
    gKern(1:model.d) =  gKern(1:model.d).*expTransform(model.kern.variance, 'gradfact');
    start = model.d;
end

gKern(start+1:end) =  gKern(start+1:end).*expTransform(model.kern.inverseWidth, 'gradfact');
 
end


