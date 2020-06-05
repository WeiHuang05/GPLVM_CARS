function [gKern, gVarmeans, gVarcovars, gInd] = kernStochasticPsi1Gradient(userId, model, kern, vardist, Z, covGrad)


M = size(Z,1); 
%da = zeros(1,model.q); du = 0; ds = 0; dz = 0;
% preallocation
gVarmeans = zeros(vardist.nParams,1);
gVarcovars = gVarmeans;
gInd = zeros(size(Z));

if model.qinverse
    gKernlengcs = zeros(1,model.q);
else
    gKernlengcs = zeros(1,model.d);
end

if model.onevariance
    gKernvar = zeros(1,1);
else
    gKernvar = zeros(1,model.d);
end


N = model.numData(userId);
[K_fu, Knovar] = kernVardistPsi1Compute(model, kern, userId);
KfuCovGrad = K_fu.*covGrad;    
   
gKernvar(userId) = sum(sum(Knovar.*covGrad));
    
if model.qinverse
    A = kern.inverseWidth;
else
    A = kern.inverseWidth(userId);
    A = repmat(A,[1 model.q]);
end
    
Tr = model.train(model.train(:,1) == userId, :);
Xl = Tr(:,2:end-1);
S_q = zeros(1,N);
Mu_q = zeros(1,N);
    
for q=1:model.q
    if model.fiq(q) == 1
       id = Xl(:,1) + model.num_items*(q-1);
    elseif model.fiq(q) == 2
       id = Xl(:,2) + model.num_items*model.dim_item  +  model.num_context(1)*(q-model.dim_item-1);
    else
       p = model.fiq(q);
       id = Xl(:,p) + model.num_items*model.dim_item  +  sum(model.num_context(1:p-2).*model.dim_context(1:p-2)) + model.num_context(p-1)*(q-model.dim_item-sum(model.dim_context(1:p-2))-1);
    end     
        %id = model.N*(q-1)+1:model.N*q;
    S_q  = vardist.covars(id);
    Mu_q = vardist.means(id); 
    Z_q = Z(:,q)'; 
        
    % B3_q term (without A(q); see report)
    %B_q = repmat(1./(A(q)*S_q + 1), [1 M]).*(repmat(Mu_q,[1 M]) - repmat(Z_q,[N 1]));
    B_q = (repmat(Mu_q,[1 M]) - repmat(Z_q,[N 1]))./repmat(A(q)*S_q + 1, [1 M]);
    % derivatives wrt variational means and inducing inputs 
    %tmp = A(q)*((K_fu.*B_q).*covGrad);
    tmp = (B_q.*KfuCovGrad);
        
    % variational means: you sum out the columns (see report)
    gVarmeans_temp = -A(q)*sum(tmp,2);
    for n = 1 : N
        gVarmeans(id(n)) = gVarmeans(id(n))+gVarmeans_temp(n); 
    end
        
    % inducing inputs: you sum out the rows 
    gInd(:,q) = gInd(:,q)+A(q)*sum(tmp,1)'; 
    %B_q = repmat(1./(A(q)*S_q + 1), [1 M]).*dist2(Mu_q, repmat(Z_q);
    B_q = (B_q.*(repmat(Mu_q,[1 M]) - repmat(Z_q,[N 1])));
    %B1_q = -(0.5./repmat((A(q)*S_q + 1), [1 M])).*(repmat(S_q, [1 M]) + B_q);
    B1_q = (repmat(S_q, [1 M]) + B_q)./repmat((A(q)*S_q + 1), [1 M]);
        
    % gradients wrt kernel hyperparameters (lengthscales) 
    %gKernlengcs(q) = sum(sum((K_fu.*B1_q).*covGrad)); 
    gKernlengcs_q(q) =   -0.5*sum(sum(B1_q.*KfuCovGrad)); 
    gVarcovars_temp = sum((KfuCovGrad./repmat((A(q)*S_q + 1), [1 M])).*(A(q)*B_q - 1),2);
    for n = 1 : N
        gVarcovars(id(n)) = gVarcovars(id(n))+gVarcovars_temp(n)*0.5*A(q); 
        %gVarcovars(id(n)) = gVarcovars(id(n))+gVarcovars_temp(n);  wrong
    end 
    if model.qinverse
        gKernlengcs(q) = gKernlengcs(q) + gKernlengcs_q(q);
    end
end
    
if ~model.qinverse
    gKernlengcs(userId) = sum(gKernlengcs_q);
end

if model.onevariance
    gKernvar = sum(gKernvar);
end
    
gKern = [gKernvar gKernlengcs];
gInd = gInd(:)'; 
    
if model.onevariance
    gKern(1) =  gKern(1)*expTransform(model.kern.variance, 'gradfact');
    start = 1;
else
    gKern(1:model.d) =  gKern(1:model.d).*expTransform(model.kern.variance, 'gradfact');
    start = model.d;
end

gKern(start+1:end) =  gKern(start+1:end).*expTransform(model.kern.inverseWidth, 'gradfact');

end
       


