function g = vargplvmLogLikeGradients(model)

% VARGPLVMLOGLIKEGRADIENTS Compute the gradients for the variational GPLVM.
% FORMAT
% DESC returns the gradients of the log likelihood with respect to the
% parameters of the GP-LVM model and with respect to the latent
% positions of the GP-LVM model.
% ARG model : the FGPLVM structure containing the parameters and
% the latent positions.
% RETURN g : the gradients of the latent positions (or the back
% constraint's parameters) and the parameters of the GP-LVM model.
%
% FORMAT
% DESC returns the gradients of the log likelihood with respect to the
% parameters of the GP-LVM model and with respect to the latent
% positions of the GP-LVM model in seperate matrices.
% ARG model : the FGPLVM structure containing the parameters and
% the latent positions.
% RETURN gX : the gradients of the latent positions (or the back
% constraint's parameters).
% RETURN gParam : gradients of the parameters of the GP-LVM model.
%
% COPYRIGHT : Michalis K. Titsias, 2009, 2010,2011
%
% COPYRIGHT :  Mauricio Alvarez, 2009, 2010,2011
%
% COPYRIGHT :  Neil D. Lawrence, 2009, 2010,2011
%
% COPYRIGHT : Andreas Damianou, 2010,2011

% SEEALSO : vargplvmLogLikelihood, vargplvmCreate, modelLogLikeGradients

% VARGPLVM


% KL divergence terms: If there are no dynamics, the only params w.r.t to
% which we need derivatives are the var. means and covars. With dynamics
% (see below) we also need derivs. w.r.t theta_t (dyn. kernel's params).
gVarmeansKL = - model.vardist.means(:)';
% !!! the covars are optimized in the log space (otherwise the *
% becomes a / and the signs change)
gVarcovsKL = 0.5 - 0.5*model.vardist.covars(:)';

% Likelihood terms (coefficients)
[gK_uu, gPsi0, gPsi1, gPsi2, gBeta] = vargpCovGrads(model);

% Get (in three steps because the formula has three terms) the gradients of
% the likelihood part w.r.t the data kernel parameters, variational means
% and covariances (original ones). From the field model.vardist, only
% vardist.means and vardist.covars and vardist.lantentDimension are used.
[gKern1, gVarmeans1, gVarcovs1, gInd1] = kernVardistPsi1Gradient(model, model.kern, model.vardist, model.X_u, gPsi1);
[gKern2, gVarmeans2, gVarcovs2, gInd2] = kernVardistPsi2Gradient(model, model.kern, model.vardist, model.X_u, gPsi2);
[gKern0, gVarmeans0, gVarcovs0] = kernVardistPsi0Gradient(model, model.kern, model.vardist, gPsi0);
gKern3 = kernGradient(model, model.kern, model.X_u, gK_uu);

% At this point, gKern gVarmeansLik and gVarcovsLik have the derivatives for the
% likelihood part. Sum all of them to obtain the final result.
gKern = gKern0 + gKern1 + gKern2 + gKern3;

%{
model1 = model;
model1.gKern = gKern;
model1.gKern0 = gKern0;
model1.gKern1 = gKern1;
model1.gKern2 = gKern2;
model1.gKern3 = gKern3;
save ('suhi.mat', 'model1');
%}

gVarmeansLik = gVarmeans0 + gVarmeans1 + gVarmeans2;

% Allocate space for gX_u
gX_u = zeros(model.k, model.q);
for d = 1:model.d
    %%% Compute Gradients with respect to X_u %%%
    gKX = kernGradX(model, model.kern, model.X_u, model.X_u, d);
    
    % The 2 accounts for the fact that covGrad is symmetric
    gKX = gKX*2;
    dgKX = kernDiagGradX(model.kern, model.X_u);
    for i = 1:model.k
        gKX(i, :, i) = dgKX(i, :);
    end
    % Compute portion associated with gK_u
    for i = 1:model.k
        for j = 1:model.q
            gX_u1(i, j) = gKX(:, j, i)'*gK_uu{d}(:, i);
        end
    end
    gX_u = gX_u + gX_u1;
end  
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
    
        % This should work much faster
        %gX_u2 = kernKuuXuGradient(model.kern, model.X_u, gK_uu);
    
        %sum(abs(gX_u2(:)-gX_u(:)))
        %pause
    


% If we only want to exclude the derivatives for the variational
% distribution, the following big block will be skipped.

gVarcovs0 = (gVarcovs0(:).*model.vardist.covars(:))';
gVarcovs1 = (gVarcovs1(:).*model.vardist.covars(:))';
gVarcovs2 = (gVarcovs2(:).*model.vardist.covars(:))';
    
gVarcovsLik = gVarcovs0 + gVarcovs1 + gVarcovs2;
      
fw = model.KLweight;
        
gVarmeans = 2 * ( (1-fw) * gVarmeansLik + fw * gVarmeansKL');
gVarcovs = 2 * ( (1-fw) * gVarcovsLik' + fw * gVarcovsKL');
       
  

%{
    % %%% Compute Gradients with respect to X_u %%%
    % gKX = kernGradX(model.kern, model.X_u, model.X_u);
    %
    % % The 2 accounts for the fact that covGrad is symmetric
    % gKX = gKX*2;
    % dgKX = kernDiagGradX(model.kern, model.X_u);
    % for i = 1:model.k
    %   gKX(i, :, i) = dgKX(i, :);
    % end
    %
    % % Allocate space for gX_u
    % gX_u = zeros(model.k, model.q);
    % % Compute portion associated with gK_u
    % for i = 1:model.k
    %   for j = 1:model.q
    %     gX_u(i, j) = gKX(:, j, i)'*gK_uu(:, i);
    %   end
    % end
    %
    % % This should work much faster
    % %gX_u2 = kernKuuXuGradient(model.kern, model.X_u, gK_uu);
    %
    % %sum(abs(gX_u2(:)-gX_u(:)))
    % %pause
    %
    % gInd = gInd1 + gInd2 + gX_u(:)';
%}


gVar = [gVarmeans' gVarcovs'];
gParam = [gInd gKern gBeta];

g = [gVar gParam];


end


function [gK_uu, gPsi0, gPsi1, gPsi2, gBeta] = vargpCovGrads(model)
gPsi0 = zeros(1,model.d);
if model.onebeta
    gBeta = zeros(1,1);
else
    gBeta = zeros(1,model.d);
end
for i = 1:model.d
    if model.onebeta
        gK_uu{i} = 0.5 * (model.T1{i} - model.kern.beta * model.invLmT{i} * model.C{i} * model.invLm{i});
        gPsi1{i} = model.kern.beta * model.m{i} * model.B{i}';
        gPsi2{i} = (model.kern.beta/2) * model.T1{i};  
        gPsi0(i) = -0.5 * model.kern.beta;
        sigm = 1/model.kern.beta; % beta^-1
        PLm = model.invLatT{i}*model.P{i};
        tmpV = sum(sum(PLm.*PLm));
        gBeta(i) = 0.5*((model.TrC(i) + (model.numData(i)-model.k)*sigm -model.Psi0(i)) ...
                 - model.TrYY(i) + model.TrPP(i) ...
                 + (1/(model.kern.beta^2))  * sum(sum(model.invLat{i}.*model.invLat{i})) + sigm*tmpV);
    else
        gK_uu{i} = 0.5 * (model.T1{i} - model.kern.beta(i) * model.invLmT{i} * model.C{i} * model.invLm{i});
        gPsi1{i} = model.kern.beta(i) * model.m{i} * model.B{i}';
        gPsi2{i} = (model.kern.beta(i)/2) * model.T1{i}; 
        gPsi0(i) = -0.5 * model.kern.beta(i);
        sigm = 1/model.kern.beta(i); % beta^-1
        PLm = model.invLatT{i}*model.P{i};
        tmpV = sum(sum(PLm.*PLm));
        gBeta(i) = 0.5*((model.TrC(i) + (model.numData(i)-model.k)*sigm -model.Psi0(i)) ...
                 - model.TrYY(i) + model.TrPP(i) ...
                 + (1/(model.kern.beta(i)^2))  * sum(sum(model.invLat{i}.*model.invLat{i})) + sigm*tmpV);
    end
    gPsi1{i} = gPsi1{i}'; % because it is passed to "kernVardistPsi1Gradient" as gPsi1'...

end
if model.onebeta
    gBeta = sum(gBeta);
end

gBeta = gBeta.*expTransform(model.kern.beta, 'gradfact');


%g_Lambda = repmat(-0.5*model.kern.beta(i), 1, model.numData(i));
end
