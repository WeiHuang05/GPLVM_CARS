function [gKern, gVarmeans, gVarcovars, gInd] = rbfard2VardistPsi2Gradient(model, kern, vardist, Z, covGrad)


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


for i = 1: model.d 
    Tr = model.train(model.train(:,1) == i, :);
    Xl = Tr(:,2:end-1);  
    if model.qinverse
       A = kern.inverseWidth;
    else
       A = kern.inverseWidth(i);
       A = repmat(A,[1 model.q]);
    end
    N(i) = model.numData(i);
     S = zeros(N(i),Q);
    Mu = zeros(N(i),Q);
    id = zeros(N(i),Q);
    
    % evaluate the kernel matrix 
    [K, outKern, sumKern, Kgvar] = kernVardistPsi2Compute(model, kern, i);
     % gradient wrt variance of the kernel 
    gKernvar(i) = 2*sum(sum(Kgvar.*covGrad{i}));  

    % 1) line compute 0.5*(z_mq + z_m'q) for any q and store the result in a "M x Q x M" 
    %  matrix where M is the number of inducing points and Q the latent dimension
    % 2) line compute the z_mq - z_m'q, for any q
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
    asPlus1 = 2*(repmat(AQ,[N(i) 1]).*S) + 1; 
    % compute the terms a_q/(2 a_q s_nq^2 + 1), for n and q and store the result in a 
    % "N x Q" matrix
    aDasPlus1 = repmat(AQ,[N(i) 1])./asPlus1; 
    %AAA = covGrad{i};
    if model.onevariance
        covGrad{i} = (kern.variance^2)*(covGrad{i}.*outKern);
    else
        covGrad{i} = (kern.variance(i)^2)*(covGrad{i}.*outKern);
    end
    covGrad{i} = reshape(covGrad{i},[M 1 M]);
    sumKern = reshape(sumKern,[M 1 M]);
    Amq = repmat(AQ,[M 1]);
       
    partInd1 = - Amq.*sum(ZmDZm.*repmat(sumKern.*covGrad{i},[1 Q 1]),3);
    partInd2 = zeros(M,Q);
 
    partA1 = - 0.25*sum(sum((ZmDZm.*ZmDZm).*repmat(sumKern.*covGrad{i},[1 Q 1]),3),1);
    partA2 = zeros(1,Q);
    
    
    % Compute the gradient wrt lengthscales, variational means and variational variances  
    % For loop over training points  
    for n = 1:N(i)
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
      k2ncovG = repmat(k2Kern_n.*covGrad{i},[1 Q 1]); 
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
      %ZmZm1 = k2kernCompute(A, mu_n, cov_n, Z); 
      %
      %AS_n = (1 + 2*A.*vardist.covars(n,:)).^0.5;  
      %
      %normfactor =  1./prod(AS_n);
      %
      %Z_n = (repmat(vardist.means(n,:),[M 1]) - Z)*0.5; 
      %Z_n = Z_n.*repmat(sqrt(A)./AS_n,[M 1]);
      %distZ = dist2(Z_n,-Z_n); 
      %
      %sumKern = sumKern + normfactor*exp(-distZ);  
      %
    end
   
    gInd = gInd + partInd1 + 2*partInd2; 
   
    if ~model.qinverse
       gKernlengcs(i) =  sum(partA1 - partA2);
    else
       gKernlengcs = gKernlengcs +  partA1 - partA2; 
    end
end

if model.onevariance
    gKernvar = sum(gKernvar);
end

gKern = [gKernvar gKernlengcs];   
gInd = gInd(:)'; 
   
end 
    
%=====================================================================
%  benchmark for gkernvariance(i), gKernlengcs(i), gInd, gvar 
%=====================================================================
%{
%for checkid = 1:25
checkid = 121;
for m1 = 1:M
for m2 = 1:M

Ps(m1,m2) = 0;
Psda(m1,m2,1) = 0;
Psda(m1,m2,2) = 0;
Psda(m1,m2,3) = 0;
Psda(m1,m2,4) = 0;
Psda(m1,m2,5) = 0;
Psda(m1,m2,6) = 0;
Psda(m1,m2,7) = 0;
Psda(m1,m2,8) = 0;
Psdu(m1,m2) = 0;
Psds(m1,m2) = 0;
Psdz(m1,m2) = 0;
for n=1:N(i)
    Psn = 1;
    Psdan = 1;
    for q = 1 : model.q

        if model.fiq(q) == 1
           id = Xl(:,1) + model.num_items*(q-1);
        elseif model.fiq(q) == 2
           id = Xl(:,2) + model.num_items*model.dim_item  +  model.num_context(1)*(q-model.dim_item-1);
        else
           p = model.fiq(q);
           id = Xl(:,p) + model.num_items*model.dim_item  +  model.num_context(1:p-2).*model.dim_context(1:p-2) + model.num_context(p-1)*(q-model.dim_item-sum(model.dim_context(1:p-2))-1);
        end   
        
         
         Sb(q)  = model.vardist.covars(id1);
         Mb(q) = model.vardist.means(id1);
         Qb = (2*A(q)*Sb(q)+1);
         Qbda = 2*Sb(q);
         Qbdu = 0;
         Qbds = 2*A(q);
            
         Bb1 = Qb^0.5;
         Bb1da = 0.5*Qb^(-0.5)*Qbda;
         Bb1du = 0;
         Bb1ds = 0;
         if(id1 == checkid)
            Bb1ds = 0.5*Qb^(-0.5)*Qbds;
         end
         
         Dz = Z(m1,q)-Z(m2,q);
         Ab1 = -A(q)*Dz^2/4;
         Ab1da = -Dz^2/4;
         Ab1du = 0;
         Ab1ds = 0;
         Ab1dz = 0;
         if (m1==2 && q == 1)
           Ab1dz = -A(q)*2*Dz/4;
         end
         if (m2==2 && q == 1)
           Ab1dz = Ab1dz+A(q)*2*Dz/4;
         end
         
         
         PZ = Mb(q)-(Z(m1,q)+Z(m2,q))/2;
         Ab2 = -A(q)*PZ^2/Qb;
         Ab2da = -(PZ^2*Qb-A(q)*PZ^2*Qbda)/Qb^2;
         Ab2du = 0;
         Ab2ds = 0;
         if(id1 == checkid)
                  Ab2du = -A(q)*2*PZ/Qb;
                  Ab2ds = 2*A(q)^2*PZ^2/Qb^2;
         end
         Ab2dz = 0;
         if (m1==2 && q == 1)
           Ab2dz = A(q)*PZ/Qb;
         end
         if (m2==2 && q == 1)
           Ab2dz = Ab2dz+A(q)*PZ/Qb;
         end
          
         Psq(q) = exp(Ab1+Ab2)/Bb1;
         Psdaq(q) = (Bb1*exp(Ab1+Ab2)*(Ab1da+Ab2da)-exp(Ab1+Ab2)*Bb1da)/Bb1^2;
         Psduq(q) = exp(Ab1+Ab2)/Bb1*Ab2du;    
         Psdsq(q) = (Bb1*exp(Ab1+Ab2)*(Ab1ds+Ab2ds)-exp(Ab1+Ab2)*Bb1ds)/Bb1^2;
         Psdzq(q) = exp(Ab1+Ab2)/Bb1*(Ab1dz+Ab2dz);    

       %  Psn = Psn*Psq;
       %  Psdan = Psdan*Psdaq;
    end
    Ps(m1,m2) = Ps(m1,m2)+Psq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,1) = Psda(m1,m2,1)+Psdaq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,2) = Psda(m1,m2,2)+Psdaq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,3) = Psda(m1,m2,3)+Psdaq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,4) = Psda(m1,m2,4)+Psdaq(4)*Psq(2)*Psq(3)*Psq(1)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,5) = Psda(m1,m2,5)+Psdaq(5)*Psq(1)*Psq(3)*Psq(4)*Psq(2)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,6) = Psda(m1,m2,6)+Psdaq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8);
    Psda(m1,m2,7) = Psda(m1,m2,7)+Psdaq(7)*Psq(1)*Psq(3)*Psq(4)*Psq(2)*Psq(6)*Psq(8)*Psq(5);
    Psda(m1,m2,8) = Psda(m1,m2,8)+Psdaq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(6);
    

    Psdu(m1,m2) = Psdu(m1,m2)+Psduq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psduq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                     +Psduq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psduq(4)*Psq(2)*Psq(1)*Psq(3)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                       +Psduq(5)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(7)*Psq(8)+Psduq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8)...
                         +Psduq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(8)+Psduq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(7);
                     
    Psds(m1,m2) = Psds(m1,m2)+Psdsq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdsq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                     +Psdsq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdsq(4)*Psq(2)*Psq(1)*Psq(3)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                       +Psdsq(5)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(7)*Psq(8)+Psdsq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8)...
                         +Psdsq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(8)+Psdsq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(7);
                     
    Psdz(m1,m2) = Psdz(m1,m2)+Psdzq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdzq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                     +Psdzq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdzq(4)*Psq(2)*Psq(1)*Psq(3)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                       +Psdzq(5)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(7)*Psq(8)+Psdzq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8)...
                         +Psdzq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(8)+Psdzq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(7);
    

    
end
Ps(m1,m2) = Ps(m1,m2)*kern.variance(i)*kern.variance(i);
Psda(m1,m2,1) = Psda(m1,m2,1)*kern.variance(i)*kern.variance(i);
Psda(m1,m2,2) = Psda(m1,m2,2)*kern.variance(i)*kern.variance(i);
Psda(m1,m2,3) = Psda(m1,m2,3)*kern.variance(i)*kern.variance(i);
Psda(m1,m2,4) = Psda(m1,m2,4)*kern.variance(i)*kern.variance(i);
Psda(m1,m2,5) = Psda(m1,m2,5)*kern.variance(i)*kern.variance(i);
Psda(m1,m2,6) = Psda(m1,m2,6)*kern.variance(i)*kern.variance(i);
Psda(m1,m2,7) = Psda(m1,m2,7)*kern.variance(i)*kern.variance(i);
Psda(m1,m2,8) = Psda(m1,m2,8)*kern.variance(i)*kern.variance(i);
Psdu(m1,m2) = Psdu(m1,m2)*kern.variance(i)*kern.variance(i);
Psds(m1,m2) = Psds(m1,m2)*kern.variance(i)*kern.variance(i);
Psdz(m1,m2) = Psdz(m1,m2)*kern.variance(i)*kern.variance(i);
end
end
2*sum(sum(Ps.*AAA))/model.kern.variance(i)-gKernvar(i)

for q = 1 : model.q
    da(q) = da(q) + sum(sum(Psda(:,:,q).*AAA));
end

da-gKernlengcs
du = du+sum(sum(Psdu.*AAA'));
ds = ds+sum(sum(Psds.*AAA'));
dz = dz+sum(sum(Psdz.*AAA'));
du-gVarmeans(checkid)
ds-gVarcovars(checkid)
dz-gInd(2,1)

%end
%}


% gVarmeans is N x Q matrix (N:number of data, Q:latent dimension)
% this will unfold this matrix column-wise 
%gVarmeans = gVarmeans'; 
%gVarmeans = gVarmeans(:)'; 

% gVarcovars is N x Q matrix (N:number of data, Q:latent dimension)
% this will unfold this matrix column-wise 
%gVarcovars = gVarcovars'; 
%gVarcovars = gVarcovars(:)';

% gInd is M x Q matrix (M:number of inducing variables, Q:latent dimension)
% this will unfold this matrix column-wise 
%gInd = gInd'; 





%{
function [gKern, gVarmeans, gVarcovars, gInd, precomp] = rbfard2VardistPsi2GradientPrecomp(rbfardKern, vardist, Z, covGrad, learnInducing)
% In the scenario here we have:
% vardist.means -> FIXED
% vardist.covars -> ~ 0
% Z -> FIXED
% Gradients w.r.t means and covars are 0

% variational means
N = size(vardist.means,1);
%  inducing variables 
[M Q] = size(Z); 


% evaluate the kernel matrix 
[K, outKern, sumKern, Kgvar] = rbfard2VardistPsi2Compute(rbfardKern, vardist, Z);

% inverse variances
A = rbfardKern.inputScales;

% gradient wrt variance of the kernel 
gKernvar = 2*sum(sum(Kgvar.*covGrad));  


% 1) line compute 0.5*(z_mq + z_m'q) for any q and store the result in a "M x Q x M" 
%  matrix where M is the number of inducing points and Q the latent dimension
% 2) line compute the z_mq - z_m'q, for any q
ZmZm  = zeros(M,Q,M);
ZmDZm = zeros(M,Q,M);
for q=1:size(Z,2)
  ZmZm(:,q,:) = 0.5*(repmat(Z(:,q),[1 1 M]) + repmat(reshape(Z(:,q),[1 1 M]),[M 1 1]));
  ZmDZm(:,q,:) = repmat(Z(:,q),[1 1 M]) - repmat(reshape(Z(:,q),[1 1 M]),[M 1 1]);
end

% compute the terms 2 a_q s_nq^2 + 1, for n and q and srore the result in a 
% "N x Q" matrix
asPlus1 = 2*(repmat(A,[N 1]).*vardist.covars) + 1;  % That's just ones
% compute the terms a_q/(2 a_q s_nq^2 + 1), for n and q and store the result in a 
% "N x Q" matrix
aDasPlus1 = repmat(A,[N 1])./asPlus1;  % That's just A repmat'ed


% CONTINUE




covGrad = (rbfardKern.variance^2)*(covGrad.*outKern);
covGrad = reshape(covGrad,[M 1 M]);
sumKern = reshape(sumKern,[M 1 M]);
Amq = repmat(A,[M 1]);
if learnInducing
    partInd1 = - Amq.*sum(ZmDZm.*repmat(sumKern.*covGrad,[1 Q 1]),3);
    partInd2 = zeros(M,Q);
end

partA1 = - 0.25*sum(sum((ZmDZm.*ZmDZm).*repmat(sumKern.*covGrad,[1 Q 1]),3),1);
partA2 = zeros(1,Q);

gVarcovars = zeros(N,Q); 
gVarmeans = zeros(N,Q);

% Compute the gradient wrt lengthscales, variational means and variational variances  
% For loop over training points  
for n=1:N
    %
    %  
    mu_n = vardist.means(n,:); 
    s2_n = vardist.covars(n,:); 
    AS_n = asPlus1(n,:);  
     
    %MunZmZm = repmat(mu_n, [M 1 M]) - ZmZm;
    MunZmZm = bsxfun(@minus,mu_n,ZmZm);
    %MunZmZmA = MunZmZm./repmat(AS_n,[M 1 M]);
    MunZmZmA =  bsxfun(@rdivide, MunZmZm, AS_n);
    
    %k2Kern_n = sum((MunZmZm.^2).*repmat(aDasPlus1(n,:),[M 1 M]),2);    
    k2Kern_n = sum(  bsxfun(@times, MunZmZm.^2,aDasPlus1(n,:)),2);
    
    k2Kern_n = exp(-k2Kern_n)/prod(sqrt(AS_n));
    
    % derivatives wrt to variational means
    k2ncovG = repmat(k2Kern_n.*covGrad,[1 Q 1]); 
    %tmp2 = tmp + reshape(diag(diag(squeeze(tmp))),[M 1 M]);
    %diagCorr = diag(diag(squeeze(tmp))); 
    tmp = MunZmZmA.*k2ncovG;
    tmp = sum(tmp,3);
    gVarmeans(n,:) = - 2*A.*(sum(tmp,1));
    
    % derivatives wrt inducing inputs 
    %diagCorr = diagCorr*(repmat(mu_n,[M 1]) - Z).*repmat(aDasPlus1(n,:),[M 1]);
    %partInd2 = partInd2 + Amq.*(sum(tmp,3) + diagCorr);
    if learnInducing
        partInd2 = partInd2 + Amq.*tmp;
    end
    
    
    % Derivative wrt input scales  
    MunZmZmA = MunZmZmA.*MunZmZm; 
    %partA2 = partA2 + sum(sum(((MunZmZmA + repmat(s2_n,[M 1 M])).*k2ncovG)./repmat(AS_n,[M 1 M]),1),3);
    tmppartA2 = bsxfun(@plus, MunZmZmA,s2_n).*k2ncovG;
    partA2 = partA2 + sum(sum( bsxfun(@rdivide, tmppartA2, AS_n), 1),3);
    
    % derivatives wrt variational diagonal covariances 
    %MunZmZmA = MunZmZmA.*repmat(A,[M 1 M]);
    MunZmZmA = bsxfun(@times, MunZmZmA, A);
    %gVarcovars(n,:) = sum(sum(repmat(aDasPlus1(n,:),[M 1 M]).*(2*MunZmZmA - 1).*k2ncovG,1),3);
    gVarcovars(n,:) = sum(sum( bsxfun(@times, (2*MunZmZmA - 1).*k2ncovG, aDasPlus1(n,:)),1),3);
    
    %ZmZm1 = k2kernCompute(A, mu_n, cov_n, Z); 
    %
    %AS_n = (1 + 2*A.*vardist.covars(n,:)).^0.5;  
    %
    %normfactor =  1./prod(AS_n);
    %
    %Z_n = (repmat(vardist.means(n,:),[M 1]) - Z)*0.5; 
    %Z_n = Z_n.*repmat(sqrt(A)./AS_n,[M 1]);
    %distZ = dist2(Z_n,-Z_n); 
    %
    %sumKern = sumKern + normfactor*exp(-distZ);  
    %
end

if learnInducing
    gInd = partInd1 + 2*partInd2; 
else
    gInd = zeros(M,Q);
end

gKernlengcs = partA1 - partA2; 
gKern = [gKernvar gKernlengcs];

% gVarmeans is N x Q matrix (N:number of data, Q:latent dimension)
% this will unfold this matrix column-wise 
%gVarmeans = gVarmeans'; 
gVarmeans = gVarmeans(:)'; 

% gVarcovars is N x Q matrix (N:number of data, Q:latent dimension)
% this will unfold this matrix column-wise 
%gVarcovars = gVarcovars'; 
gVarcovars = gVarcovars(:)';

% gInd is M x Q matrix (M:number of inducing variables, Q:latent dimension)
% this will unfold this matrix column-wise 
%gInd = gInd'; 
gInd = gInd(:)'; 
%}
