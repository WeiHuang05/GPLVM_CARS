function [Psi1, Knovar] = kernVardistPsi1Compute(model, kern, i)


Z = model.X_u;
N  = model.numData(i);
M = size(Z,1); 

if model.qinverse
    A = kern.inverseWidth;
else
    A = kern.inverseWidth(i);
    A = repmat(A,[1 model.q]);
end

argExp = zeros(N,M); 
normfactor = ones(N,1);
Tr = model.train(model.train(:,1) == i, :);
Xl = Tr(:,2:end-1);

for q=1:model.q
%    
    if model.fiq(q) == 1
        id = Xl(:,1) + model.num_items*(q-1);
    elseif model.fiq(q) == 2
        id = Xl(:,2) + model.num_items*model.dim_item  +  model.num_context(1)*(q-model.dim_item-1);
    else
        p = model.fiq(q);
        id = Xl(:,p) + model.num_items*model.dim_item  +  sum(model.num_context(1:p-2).*model.dim_context(1:p-2)) + model.num_context(p-1)*(q-model.dim_item-sum(model.dim_context(1:p-2))-1);
    end     
 
    %id = model.N*(q-1)+1:model.N*q;
    S_q  = model.vardist.covars(id);
    Mu_q = model.vardist.means(id); 
    normfactor = normfactor.*(A(q)*S_q + 1);
    Z_q = Z(:,q)';
  
    distan = (repmat(Mu_q,[1 M]) - repmat(Z_q,[N 1])).^2;    
    argExp = argExp + repmat(A(q)./(A(q)*S_q + 1), [1 M]).*distan;            
%
end
normfactor = normfactor.^0.5;
Knovar = repmat(1./normfactor,[1 M]).*exp(-0.5*argExp); 
if model.onevariance
   Psi1 = kern.variance*Knovar; 
else
   Psi1 = kern.variance(i)*Knovar; 
end
%========================================================
%  benchmark method
%========================================================
%{
for n = 1:N
for m = 1:M
    Ps(n,m) = 1;
    for q = 1 : model.q
         if q==1
            id = Xl(n,q);
         elseif q == 2
            id = Xl(n,q)+model.num_items;
         else
            id = Xl(n,q)+model.num_items+sum(model.num_context(1:q-2));
         end                                                 
         S(q)  = model.vardist.covars(id);
         Mu(q) = model.vardist.means(id);
         Q = A(q)*S(q)+1;
         B1 = (A(q)*S(q)+1)^0.5;
         A1 = -0.5*A(q)*(Mu(q)-Z(m,q))^2/Q;
         Psq = exp(A1)/B1;
         Ps(n,m) = Ps(n,m)*Psq;
    end

Ps(n,m) = Ps(n,m)*kern.variance(i);
end
end
aaa = 1;

%}

