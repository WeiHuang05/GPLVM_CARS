function  [Psi2, outKern, sumKern, Kgvar] = kernVardistPsi2Compute(model, kern, i)

% KERNVARDISTPSI2COMPUTE description.  

% VARGPLVM
Z = model.X_u;
N  = model.numData(i);
M = size(Z,1); 
if model.qinverse
    A = kern.inverseWidth;
else
    A = kern.inverseWidth(i);
    A = repmat(A,[1 model.q]);
end
  
% first way
sumKern = zeros(M,M); 
Tr = model.train(model.train(:,1) == i, :);
Xl = Tr(:,2:end-1);

%========================================================
for n=1:N
    % 
    for q = 1 : model.q
        
         if model.fiq(q) == 1
            id = Xl(n,1) + model.num_items*(q-1);
         elseif model.fiq(q) == 2
            id = Xl(n,2) + model.num_items*model.dim_item  +  model.num_context(1)*(q-model.dim_item-1);
         else
            p = model.fiq(q);
            id = Xl(n,p) + model.num_items*model.dim_item  +  sum(model.num_context(1:p-2).*model.dim_context(1:p-2)) + model.num_context(p-1)*(q-model.dim_item-sum(model.dim_context(1:p-2))-1);
         end     
        
         %id = model.N*(q-1)+n;
         S_n(q)  = model.vardist.covars(id);
         Mu_n(q) = model.vardist.means(id); 
    end
        
    %AS_n = (1 + 2*A.*vardist.covars(n,:)).^0.5;  
    AS_n = (1 + 2*A.*S_n).^0.5;  
    normfactor =  1./prod(AS_n);
    
    %Z_n = (repmat(vardist.means(n,:),[M 1]) - Z)*0.5; 
    Z_n = bsxfun(@minus, Mu_n, Z)*0.5;
    %Z_n = Z_n.*repmat(sqrt(A)./AS_n,[M 1]);
    Z_n = bsxfun(@times, Z_n, sqrt(A)./AS_n);
    distZ = dist2(Z_n,-Z_n); 
    
    sumKern = sumKern + normfactor*exp(-distZ);  
    %
end
    
% ZZ = Z.*(repmat(sqrt(A),[M 1]));
ZZ =  bsxfun(@times, Z, sqrt(A));
distZZ = dist2(ZZ,ZZ);
outKern = exp(-0.25*distZZ);

if model.onevariance
   Kgvar = kern.variance*(outKern.*sumKern); 
   Psi2 = kern.variance*Kgvar;
else
   Kgvar = kern.variance(i)*(outKern.*sumKern); 
   Psi2 = kern.variance(i)*Kgvar;
end

%========================================================
%  benchmark method
%========================================================
%{
m1 = 1;
m2 = 2;

Ps = 0;
for n=1:N
    Psn = 1;
    for q = 1 : model.q
        if model.fiq(q) == 1
            id = Xl(:,1) + model.num_items*(q-1);
         elseif model.fiq(q) == 2
            id = Xl(:,2) + model.num_items*model.dim_item  +  model.num_context(1)*(q-model.dim_item-1);
         else
            p = model.fiq(q);
           id = Xl(:,p) + model.num_items*model.dim_item  +  model.num_context(1:p-2).*model.dim_context(1:p-2) + model.num_context(p-1)*(q-model.dim_item-sum(model.dim_context(1:p-2))-1);
         end     
         
         S(q)  = model.vardist.covars(id);
         M(q) = model.vardist.means(id);
         Q = 2*A(q)*S(q)+1;
         B1 = (2*A(q)*S(q)+1)^0.5;
         A1 = -A(q)*(Z(m1,q)-Z(m2,q))^2/4;
         A2 = -A(q)*(M(q)-(Z(m1,q)+Z(m2,q))/2)^2/Q;
         Psq = exp(A1+A2)/Q^0.5;
         Psn = Psn*Psq;
    end
    Ps = Ps+Psn;
end
Ps = Ps*kern.variance(i)*kern.variance(i);


%}

    

