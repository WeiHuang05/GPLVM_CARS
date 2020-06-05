function [gKern, gVarmeans, gVarcovars, gInd] = rbfard2VardistPsi1Gradient(model, kern, vardist, Z, covGrad)


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


for i = 1: model.d
   
    N(i) = model.numData(i);
    [K_fu, Knovar] = kernVardistPsi1Compute(model, kern, i);
    KfuCovGrad = K_fu.*covGrad{i}';    
    % gradient wrt variance of the kernel 
    gKernvar(i) = sum(sum(Knovar.*covGrad{i}'));
    
    if model.qinverse
       A = kern.inverseWidth;
    else
       A = kern.inverseWidth(i);
       A = repmat(A,[1 model.q]);
    end
    
    Tr = model.train(model.train(:,1) == i, :);
    Xl = Tr(:,2:end-1);
    S_q = zeros(1,N(i));
    Mu_q = zeros(1,N(i));
    
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
        B_q = (repmat(Mu_q,[1 M]) - repmat(Z_q,[N(i) 1]))./repmat(A(q)*S_q + 1, [1 M]);
        % derivatives wrt variational means and inducing inputs 
        %tmp = A(q)*((K_fu.*B_q).*covGrad);
        tmp = (B_q.*KfuCovGrad);
        
        % variational means: you sum out the columns (see report)
        gVarmeans_temp = -A(q)*sum(tmp,2);
        for n = 1 : N(i)
            gVarmeans(id(n)) = gVarmeans(id(n))+gVarmeans_temp(n); 
        end
        
        % inducing inputs: you sum out the rows 
        gInd(:,q) = gInd(:,q)+A(q)*sum(tmp,1)'; 
        %B_q = repmat(1./(A(q)*S_q + 1), [1 M]).*dist2(Mu_q, repmat(Z_q);
        B_q = (B_q.*(repmat(Mu_q,[1 M]) - repmat(Z_q,[N(i) 1])));
        %B1_q = -(0.5./repmat((A(q)*S_q + 1), [1 M])).*(repmat(S_q, [1 M]) + B_q);
        B1_q = (repmat(S_q, [1 M]) + B_q)./repmat((A(q)*S_q + 1), [1 M]);
        
        % gradients wrt kernel hyperparameters (lengthscales) 
        %gKernlengcs(q) = sum(sum((K_fu.*B1_q).*covGrad)); 
        gKernlengcs_q(q) =   -0.5*sum(sum(B1_q.*KfuCovGrad)); 
        gVarcovars_temp = sum((KfuCovGrad./repmat((A(q)*S_q + 1), [1 M])).*(A(q)*B_q - 1),2);
        for n = 1 : N(i)
            gVarcovars(id(n)) = gVarcovars(id(n))+gVarcovars_temp(n)*0.5*A(q); 
          %gVarcovars(id(n)) = gVarcovars(id(n))+gVarcovars_temp(n);  wrong
        end 
        if model.qinverse
            gKernlengcs(q) = gKernlengcs(q) + gKernlengcs_q(q);
        end
    end
    
    if ~model.qinverse
        gKernlengcs(i) = sum(gKernlengcs_q);
    end
end 

if model.onevariance
    gKernvar = sum(gKernvar);
end
    
gKern = [gKernvar gKernlengcs];
gInd = gInd(:)'; 
%=====================================================================
%  benchmark for gkernvariance(i), gKernlengcs(i), gInd, gvar 
%=====================================================================
%{   
%for checkid = 1:25
checkid = 121;
for n = 1:N(i)
for m = 1:M

Ps(n,m) = 0;
%Psda(n,m) = 0;
Psdu(n,m) = 0;
Psds(n,m) = 0;
Psdz(n,m) = 0;

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
         Qb = (A(q)*Sb(q)+1);
         Qbda = Sb(q);
         Qbdu = 0;
         Qbds = A(q);
            
         Bb1 = Qb^0.5;
         Bb1da = 0.5*Qb^(-0.5)*Qbda;
         Bb1du = 0;
         Bb1ds = 0;
         if(id1 == checkid)
            Bb1ds = 0.5*Qb^(-0.5)*Qbds;
         end
         
         Dz = Mb(q)-Z(m,q);
         Ab1 = -0.5*A(q)*Dz^2/Qb;
         Ab1da = -0.5*(Dz^2*Qb-Qbda*A(q)*Dz^2)/Qb^2;
         Ab1du = 0;
         Ab1ds = 0;
         Ab1dz = 0;
         if(id1 == checkid)
            Ab1du = -0.5*A(q)*2*Dz/Qb;
            Ab1ds = 0.5*A(q)*Dz^2*Qbds/Qb^2;
         end
         if (m==2 && q == 1)
           Ab1dz = 0.5*A(q)*2*Dz/Qb;
         end
                        
         Psq(q) = exp(Ab1)/Bb1;
         Psdaq(q) = (Bb1*exp(Ab1)*Ab1da-exp(Ab1)*Bb1da)/Bb1^2;
         Psduq(q) = exp(Ab1)/Bb1*Ab1du;    
         Psdsq(q) = (Bb1*exp(Ab1)*Ab1ds-exp(Ab1)*Bb1ds)/Bb1^2;
         Psdzq(q) = exp(Ab1)/Bb1*Ab1dz;    

       %  Psn = Psn*Psq;
       %  Psdan = Psdan*Psdaq;
    end
    Ps(n,m) = Psq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(n,m,1) = Psdaq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(n,m,2) = Psdaq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(n,m,3) = Psdaq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(n,m,4) = Psdaq(4)*Psq(2)*Psq(3)*Psq(1)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(n,m,5) = Psdaq(5)*Psq(1)*Psq(3)*Psq(4)*Psq(2)*Psq(6)*Psq(7)*Psq(8);
    Psda(n,m,6) = Psdaq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8);
    Psda(n,m,7) = Psdaq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(6)*Psq(8);
    Psda(n,m,8) = Psdaq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(6);
    
    Psdu(n,m) = Psduq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psduq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                     +Psduq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psduq(4)*Psq(2)*Psq(1)*Psq(3)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                       +Psduq(5)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(7)*Psq(8)+Psduq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8)...
                         +Psduq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(8)+Psduq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(7);
                     
    Psds(n,m) = Psdsq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdsq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                     +Psdsq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdsq(4)*Psq(2)*Psq(1)*Psq(3)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                       +Psdsq(5)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(7)*Psq(8)+Psdsq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8)...
                         +Psdsq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(8)+Psdsq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(7);
                                                                                  
    Psdz(n,m) = Psdzq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdzq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                     +Psdzq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdzq(4)*Psq(2)*Psq(1)*Psq(3)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                       +Psdzq(5)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(7)*Psq(8)+Psdzq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8)...
                         +Psdzq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(8)+Psdzq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(7);
    
Ps(n,m) = Ps(n,m)*kern.variance(i);
Psda(n,m,1) = Psda(n,m,1)*kern.variance(i);
Psda(n,m,2) = Psda(n,m,2)*kern.variance(i);
Psda(n,m,3) = Psda(n,m,3)*kern.variance(i);
Psda(n,m,4) = Psda(n,m,4)*kern.variance(i);
Psda(n,m,5) = Psda(n,m,5)*kern.variance(i);
Psda(n,m,6) = Psda(n,m,6)*kern.variance(i);
Psda(n,m,7) = Psda(n,m,7)*kern.variance(i);
Psda(n,m,8) = Psda(n,m,8)*kern.variance(i);
Psdu(n,m) = Psdu(n,m)*kern.variance(i);
Psds(n,m) = Psds(n,m)*kern.variance(i);
Psdz(n,m) = Psdz(n,m)*kern.variance(i);
end
end

for q = 1 : model.q
    da(q) = da(q) + sum(sum(Psda(1:N(i),:,q).*covGrad{i}'));
end
sum(sum(Ps(1:N(i),:).*covGrad{i}'/kern.variance(i)))-gKernvar(i)
da-gKernlengcs
du = du+sum(sum(Psdu(1:N(i),:).*covGrad{i}'));
ds = ds+sum(sum(Psds(1:N(i),:).*covGrad{i}'));
dz = dz+sum(sum(Psdz(1:N(i),:).*covGrad{i}'));
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
%gVarcovars = 0.5*repmat(A,[N 1]).*gVarcovars;
%gVarcovars = gVarcovars(:)';

% gInd is M x Q matrix (M:number of inducing variables, Q:latent dimension)
% this will unfold this matrix column-wise 
%gInd = gInd'; 


