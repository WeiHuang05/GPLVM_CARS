function [K, Knovar, argExp] = rbfard2VardistPsi1Compute(userId, model, Kern, vardist, Z)


N  = size(vardist.means,1);
M = size(Z,1); 
Q = size(Z,2);

if model.qinverse
   A = Kern.inverseWidth;
else
   A = Kern.inverseWidth(userId);
   A = repmat(A,[1,Q]);
end

         
argExp = zeros(N,M); 
normfactor = ones(N,1);
for q=1:model.q
%
    S_q = vardist.covars(:,q);  
    normfactor = normfactor.*(A(q)*S_q + 1);
    Mu_q = vardist.means(:,q); 
    Z_q = Z(:,q)';
    distan = (repmat(Mu_q,[1 M]) - repmat(Z_q,[N 1])).^2;
    argExp = argExp + repmat(A(q)./(A(q)*S_q + 1), [1 M]).*distan;
%
end
normfactor = normfactor.^0.5;
Knovar = repmat(1./normfactor,[1 M]).*exp(-0.5*argExp); 
if model.onevariance
   K = Kern.variance*Knovar; 
else
   K = Kern.variance(userId)*Knovar; 
end

