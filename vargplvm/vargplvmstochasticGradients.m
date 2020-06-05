function g = vargplvmstochasticGradients(userId, model)


gVarmeansKL = - model.vardist.means(:)';
gVarcovsKL = 0.5 - 0.5*model.vardist.covars(:)';

gVarmeansKL = gVarmeansKL/model.num_users;
gVarcovsKL  = gVarcovsKL/model.num_users;

[gK_uu, gPsi0, gPsi1, gPsi2, gBeta] = vargpCovGrads(userId, model);


[gKern1, gVarmeans1, gVarcovs1, gInd1] = kernStochasticPsi1Gradient(userId, model, model.kern, model.vardist, model.X_u, gPsi1);
[gKern2, gVarmeans2, gVarcovs2, gInd2] = kernStochasticPsi2Gradient(userId, model, model.kern, model.vardist, model.X_u, gPsi2);
[gKern0, gVarmeans0, gVarcovs0] = kernStochasticPsi0Gradient(userId, model, model.kern, model.vardist, gPsi0);
gKern3 = kernStochasticGradient(userId, model, model.kern, model.X_u, gK_uu);


gKern = gKern0 + gKern1 + gKern2 + gKern3;

gVarmeansLik = gVarmeans0 + gVarmeans1 + gVarmeans2;

gX_u = zeros(model.k, model.q);

gKX = kernGradX(model, model.kern, model.X_u, model.X_u, userId);
    
gKX = gKX*2;
dgKX = kernDiagGradX(model.kern, model.X_u);
for i = 1:model.k
    gKX(i, :, i) = dgKX(i, :);
end
  
for i = 1:model.k
     for j = 1:model.q
         gX_u1(i, j) = gKX(:, j, i)'*gK_uu(:, i);
     end
end
gX_u = gX_u + gX_u1;
  
gX_u = gX_u(:)';
   
gInd = gInd1 + gInd2 + gX_u;
  
        
%=====================================================================
%  benchmark for gInd
%=====================================================================
%{
A = model.kern.inverseWidth;
M = size(model.X_u,1); 
x = model.X_u;
a = 2;
b = 1;
for m1 = 1:M
for m2 = 1:M

Ps(m1,m2) = 0;
Psdx(m1,m2) = 0;

    for q = 1 : model.q

         Dz = x(m1,q)-x(m2,q);
         Ab = -0.5*A(q)*Dz^2;
         Abdx = 0;
         if (m1==a && q==b)
             Abdx = -A(q)*Dz;
         end
         if (m2==a && q==b)
             Abdx = Abdx+A(q)*Dz;
         end
                   
         Psq(q) = exp(Ab);
         Psdxq(q) =exp(Ab)*Abdx;
      
    end
    Ps(m1,m2) = Psq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8);
    Psdx(m1,m2) = Psdxq(1)*Psq(2)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdxq(2)*Psq(1)*Psq(3)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                     +Psdxq(3)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(6)*Psq(7)*Psq(8)+Psdxq(4)*Psq(2)*Psq(1)*Psq(3)*Psq(5)*Psq(6)*Psq(7)*Psq(8)...
                       +Psdxq(5)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(7)*Psq(8)+Psdxq(6)*Psq(2)*Psq(1)*Psq(4)*Psq(5)*Psq(3)*Psq(7)*Psq(8)...
                         +Psdxq(7)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(8)+Psdxq(8)*Psq(2)*Psq(1)*Psq(4)*Psq(3)*Psq(6)*Psq(5)*Psq(7);
    
    
Ps(m1,m2) = Ps(m1,m2)*model.kern.variance(d);
Psdx(m1,m2) = Psdx(m1,m2)*model.kern.variance(d);


end
end
sum(sum(Ps.*gK_uu{d}/model.kern.variance(d)))-gKern3(d)/log(model.kern.variance(d))
sum(sum(Psdx.*gK_uu{d}))-gX_u1(a,b)

%end
%}    


gVarcovs0 = (gVarcovs0(:).*model.vardist.covars(:))';
gVarcovs1 = (gVarcovs1(:).*model.vardist.covars(:))';
gVarcovs2 = (gVarcovs2(:).*model.vardist.covars(:))';
    
gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2;
      
fw = model.KLweight;
        
gVarmeans = 2 * ( (1-fw) * gVarmeansLik + fw * gVarmeansKL');
gVarcovs = 2 * ( (1-fw) * gVarcovsLik' + fw * gVarcovsKL');
       

gVar = [gVarmeans' gVarcovs'];
gParam = [gInd gKern gBeta];

g = [gVar gParam];


end


function [gK_uu, gPsi0, gPsi1, gPsi2, gBeta] = vargpCovGrads(userId, model)

if model.onebeta
    gBeta = zeros(1,1);
else
    gBeta = zeros(1,model.d);
end

if model.onebeta
        gK_uu = 0.5 * (model.T1{userId} - model.kern.beta * model.invLmT{userId} * model.C{userId} * model.invLm{userId});
        gPsi1 = model.kern.beta * model.m{userId} * model.B{userId}';
        gPsi2 = (model.kern.beta/2) * model.T1{userId};  
        gPsi0 = -0.5 * model.kern.beta;
        sigm = 1/model.kern.beta; % beta^-1
        PLm = model.invLatT{userId}*model.P{userId};
        tmpV = sum(sum(PLm.*PLm));
        gBeta(userId) = 0.5*((model.TrC(userId) + (model.numData(userId)-model.k)*sigm -model.Psi0(userId)) ...
                 - model.TrYY(userId) + model.TrPP(userId) ...
                 + (1/(model.kern.beta^2))  * sum(sum(model.invLat{userId}.*model.invLat{userId})) + sigm*tmpV);
else
        gK_uu = 0.5 * (model.T1{userId} - model.kern.beta(userId) * model.invLmT{userId} * model.C{userId} * model.invLm{userId});
        gPsi1 = model.kern.beta(userId) * model.m{userId} * model.B{userId}';
        gPsi2 = (model.kern.beta(userId)/2) * model.T1{userId}; 
        gPsi0 = -0.5 * model.kern.beta(userId);
        sigm = 1/model.kern.beta(userId); % beta^-1
        PLm = model.invLatT{userId}*model.P{userId};
        tmpV = sum(sum(PLm.*PLm));
        gBeta(userId) = 0.5*((model.TrC(userId) + (model.numData(userId)-model.k)*sigm -model.Psi0(userId)) ...
                 - model.TrYY(userId) + model.TrPP(userId) ...
                 + (1/(model.kern.beta(userId)^2))  * sum(sum(model.invLat{userId}.*model.invLat{userId})) + sigm*tmpV);
end

if model.onebeta
    gBeta = sum(gBeta);
end

gBeta = gBeta.*expTransform(model.kern.beta, 'gradfact');


end
