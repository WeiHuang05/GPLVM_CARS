function [gKern, gVarmeans, gVarcovars] = rbfard2VardistPsi0Gradient(model, kern, vardist, covGrad)

% RBFARD2VARDISTPSI0GRADIENT Description
  
% VARGPLVM
for i = 1:model.d
    gKernvar(i) = covGrad(i)*model.numData(i);  
end

if model.onevariance
    gKernvar = sum(gKernvar);
end

if model.qinverse
   gKernlengcs = zeros(1,model.q);
else
   gKernlengcs = zeros(1,model.d);
end

gKern = [gKernvar gKernlengcs];

gVarmeans = zeros(vardist.nParams,1); 
gVarcovars = zeros(vardist.nParams,1); 




