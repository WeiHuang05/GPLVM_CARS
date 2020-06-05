function [gKern, gVarmeans, gVarcovars, gInd] = kernStochasticPsi2Gradient(userId, model, kern, vardist, Z, covGrad)

%  inducing variables 
[M, Q] = size(Z); 
gVarmeans = zeros(vardist.nParams,1);
gVarcovars = gVarmeans;
gInd = zeros(size(Z));
%da = zeros(1,model.q);  du = 0;  ds = 0;  dz = 0;

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

Tr = model.train(model.train(:,1) == userId, :);
Xl = Tr(:,2:end-1);  
if model.qinverse
    A = kern.inverseWidth;
else
    A = kern.inverseWidth(i);
    A = repmat(A,[1 model.q]);
end
N = model.numData(userId);
S = zeros(N,Q);
Mu = zeros(N,Q);
id = zeros(N,Q);
    
% evaluate the kernel matrix 
[K, outKern, sumKern, Kgvar] = kernVardistPsi2Compute(model, kern, userId);
% gradient wrt variance of the kernel 
gKernvar(userId) = 2*sum(sum(Kgvar.*covGrad));  


ZmZm  = zeros(M,Q,M);
ZmDZm = zeros(M,Q,M);
    
for q=1:size(Z,2)
    ZmZm(:,q,:) = 0.5*(repmat(Z(:,q),[1 1 M]) + repmat(reshape(Z(:,q),[1 1 M]),[M 1 1]));
    ZmDZm(:,q,:) = repmat(Z(:,q),[1 1 M]) - repmat(reshape(Z(:,q),[1 1 M]),[M 1 1]);
     
    if model.fiq(q) == 1
        id(:,q) = Xl(:,1) + model.num_items*(q-1);
    elseif model.fiq(q) == 2
        id(:,q) = Xl(:,2) + model.num_items*model.dim_item  +  model.num_context(1)*(q-model.dim_item-1);
    else
        p = model.fiq(q);
        id(:,q) = Xl(:,p) + model.num_items*model.dim_item  +  sum(model.num_context(1:p-2).*model.dim_context(1:p-2)) + model.num_context(p-1)*(q-model.dim_item-sum(model.dim_context(1:p-2))-1);
    end   
     
    %id(:,q) = model.N*(q-1)+1:model.N*q;
    S(:,q)  = vardist.covars(id(:,q));
    Mu(:,q) = vardist.means(id(:,q)); 
          
end
% compute the terms 2 a_q s_nq^2 + 1, for n and q and srore the result in a 
% "N x Q" matrix
%AQ = repmat(A(i),[1 Q]);
AQ = A;
asPlus1 = 2*(repmat(AQ,[N 1]).*S) + 1; 
% compute the terms a_q/(2 a_q s_nq^2 + 1), for n and q and store the result in a 
% "N x Q" matrix
aDasPlus1 = repmat(AQ,[N 1])./asPlus1; 
%AAA = covGrad{i};
if model.onevariance
    covGrad = (kern.variance^2)*(covGrad.*outKern);
else
    covGrad = (kern.variance(userId)^2)*(covGrad.*outKern);
end
covGrad = reshape(covGrad,[M 1 M]);
sumKern = reshape(sumKern,[M 1 M]);
Amq = repmat(AQ,[M 1]);
       
partInd1 = - Amq.*sum(ZmDZm.*repmat(sumKern.*covGrad,[1 Q 1]),3);
partInd2 = zeros(M,Q);
 
partA1 = - 0.25*sum(sum((ZmDZm.*ZmDZm).*repmat(sumKern.*covGrad,[1 Q 1]),3),1);
partA2 = zeros(1,Q);
    

for n = 1:N
    %parfor n=1:N(i)
      S_n  = S(n,:);
      Mu_n = Mu(n,:); 
      
      AS_n = asPlus1(n,:);  
      %MunZmZm = repmat(mu_n, [M 1 M]) - ZmZm;
      MunZmZm = bsxfun(@minus,Mu_n,ZmZm);
      %MunZmZmA = MunZmZm./repmat(AS_n,[M 1 M]);
      MunZmZmA =  bsxfun(@rdivide, MunZmZm, AS_n);
    
      %k2Kern_n = sum((MunZmZm.^2).*repmat(aDasPlus1(n,:),[M 1 M]),2);    
      k2Kern_n = sum(bsxfun(@times, MunZmZm.^2,aDasPlus1(n,:)),2);
    
      k2Kern_n = exp(-k2Kern_n)/prod(sqrt(AS_n));
    
      % derivatives wrt to variational means
      k2ncovG = repmat(k2Kern_n.*covGrad,[1 Q 1]); 
      %tmp2 = tmp + reshape(diag(diag(squeeze(tmp))),[M 1 M]);
      %diagCorr = diag(diag(squeeze(tmp))); 
      tmp = MunZmZmA.*k2ncovG;
      tmp = sum(tmp,3);
      gVarmeans_temp = - 2*A.*(sum(tmp,1));
   
      gVarmeans(id(n,:)) = gVarmeans(id(n,:))+gVarmeans_temp';
    
      % derivatives wrt inducing inputs 
      %diagCorr = diagCorr*(repmat(mu_n,[M 1]) - Z).*repmat(aDasPlus1(n,:),[M 1]);
      %partInd2 = partInd2 + Amq.*(sum(tmp,3) + diagCorr);
      partInd2 = partInd2 + Amq.*tmp;
    
      % Derivative wrt input scales  
      MunZmZmA = MunZmZmA.*MunZmZm; 
      %partA2 = partA2 + sum(sum(((MunZmZmA + repmat(s2_n,[M 1 M])).*k2ncovG)./repmat(AS_n,[M 1 M]),1),3);
      tmppartA2 = bsxfun(@plus, MunZmZmA,S_n).*k2ncovG;
      partA2 = partA2 + sum(sum( bsxfun(@rdivide, tmppartA2, AS_n), 1),3);
      
      % derivatives wrt variational diagonal covariances 
      %MunZmZmA = MunZmZmA.*repmat(A,[M 1 M]);
      MunZmZmA = bsxfun(@times, MunZmZmA, AQ);
      %gVarcovars(n,:) = sum(sum(repmat(aDasPlus1(n,:),[M 1 M]).*(2*MunZmZmA - 1).*k2ncovG,1),3);
      gVarcovars_temp = sum(sum( bsxfun(@times, (2*MunZmZmA - 1).*k2ncovG, aDasPlus1(n,:)),1),3);
   
      gVarcovars(id(n,:)) = gVarcovars(id(n,:))+gVarcovars_temp'; 
     
end
   
gInd = gInd + partInd1 + 2*partInd2; 
   
if ~model.qinverse
    gKernlengcs(userId) =  sum(partA1 - partA2);
else
    gKernlengcs = gKernlengcs +  partA1 - partA2; 
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