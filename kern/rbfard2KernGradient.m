function g = rbfard2KernGradient(model, kern, x, gK_uu)


% COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006

% COPYRIGHT : Michalis K. Titsias, 2009

%da = zeros(1,model.q);
if model.qinverse
    gKernlengcs = zeros(1,model.q);
else
    gKernlengcs = zeros(1,model.d);
end

for i = 1:model.d

    [k, dist2xx] = rbfard2KernCompute(model, i);
    covGradK = gK_uu{i}.*k;
    
    if model.onevariance
       gKernvar(i) =  sum(sum(gK_uu{i}.*k))/kern.variance;
    else
       gKernvar(i) =  sum(sum(gK_uu{i}.*k))/kern.variance(i);
    end
   
    for q = 1:size(x, 2)
        gleng(q)  =  -(sum(covGradK*(x(:, q).*x(:, q))) -x(:, q)'*covGradK*x(:, q));
    end
    
    if model.qinverse
       gKernlengcs = gKernlengcs + gleng; 
    else
       gKernlengcs(i) =  sum(gleng);
    end
       
end

if model.onevariance
    gKernvar = sum(gKernvar);
end

g = [gKernvar gKernlengcs];

%=====================================================================
%  benchmark for gkernvariance(i), gKernlengcs(i)
%=====================================================================
%{
A = kern.inverseWidth;

M = size(x,1); 
for m1 = 1:M
for m2 = 1:M

Ps(m1,m2) = 0;
Psda(m1,m2) = 0;

    for q = 1 : model.q

         Dz = x(m1,q)-x(m2,q);
         Ab = -0.5*A(q)*Dz^2;
         Abda = -0.5*Dz^2;
                   
         Psq(q) = exp(Ab);
         Psdaq(q) = exp(Ab)*Abda;
      
    end
   
    Ps(m1,m2) = Psq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,1) = Psdaq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,2) = Psdaq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,3) = Psdaq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,4) = Psdaq(4)*Psq(2)*Psq(3)*Psq(1)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,5) = Psdaq(5)*Psq(1)*Psq(3)*Psq(4)*Psq(2)*Psq(6)*Psq(7)*Psq(8);
    Psda(m1,m2,6) = Psdaq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8);
    Psda(m1,m2,7) = Psdaq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(6)*Psq(8);
    Psda(m1,m2,8) = Psdaq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(6);
    
    
Ps(m1,m2) = Ps(m1,m2)*kern.variance(i);
Psda(m1,m2,1) = Psda(m1,m2,1)*kern.variance(i);
Psda(m1,m2,2) = Psda(m1,m2,2)*kern.variance(i);
Psda(m1,m2,3) = Psda(m1,m2,3)*kern.variance(i);
Psda(m1,m2,4) = Psda(m1,m2,4)*kern.variance(i);
Psda(m1,m2,5) = Psda(m1,m2,5)*kern.variance(i);
Psda(m1,m2,6) = Psda(m1,m2,6)*kern.variance(i);
Psda(m1,m2,7) = Psda(m1,m2,7)*kern.variance(i);
Psda(m1,m2,8) = Psda(m1,m2,8)*kern.variance(i);

end
end
sum(sum(Ps.*gK_uu{i}/kern.variance(i)))-gKernvar(i)
for q = 1 : model.q
    da(q) = da(q) + sum(sum(Psda(:,:,q).*gK_uu{i}));
end
da-gKernlengcs

%end
%}    
    
