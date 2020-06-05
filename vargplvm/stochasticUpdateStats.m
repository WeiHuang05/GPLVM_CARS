function model = stochasticUpdateStats(model, userId)
% VARGPLVMUPDATESTATS Update stats for VARGPLVM model.
%
% COPYRIGHT : Michalis K. Titsias, 2009-2011
% COPYRIGHT : Neil D. Lawrence, 2009-2011
% COPYRIGHT : Andreas C. Damianou, 2010-2011

  
jitter = 1e-6;
%model.jitter = 1e-6;

%%% Precomputations for (the likelihood term of) the bound %%%
     
 %   model.K_uu{i} = kernCompute(model,i);
model.K_uu{userId} = rbfard2KernCompute(model, userId);
model.K_uu{userId} = model.K_uu{userId} ...
                         + sparseDiag(repmat(jitter, size(model.K_uu{userId}, 1), 1));
 
model.Psi0(userId) = kernVardistPsi0Compute(model, model.kern,userId);
model.Psi1{userId} = kernVardistPsi1Compute(model, model.kern,userId);
model.Psi2{userId} = kernVardistPsi2Compute(model, model.kern,userId);
    
%model.Lm = chol(model.K_uu, 'lower'); 
model.Lm{userId} = jitChol(model.K_uu{userId})';      % M x M: L_m (lower triangular)   ---- O(m^3)
model.invLm{userId} = model.Lm{userId}\eye(model.k);  % M x M: L_m^{-1}                 ---- O(m^3)
model.invLmT{userId} = model.invLm{userId}'; % L_m^{-T}
model.C{userId} = model.invLm{userId} * model.Psi2{userId} * model.invLmT{userId};
model.TrC(userId) = sum(diag(model.C{userId})); % Tr(C)   Tr(ABC)=Tr(BCA)
% Matrix At replaces the matrix A of the old implementation; At is more stable
% since it has a much smaller condition number than A=sigma^2 K_uu + Psi2
if model.onebeta
   model.At{userId} = 1/model.kern.beta * eye(size(model.C{userId},1)) + model.C{userId}; % At = beta^{-1} I + C
else
   model.At{userId} = (1/model.kern.beta(userId)) * eye(size(model.C{userId},1)) + model.C{userId}; % At = beta^{-1} I + C
end
model.Lat{userId} = jitChol(model.At{userId})'; % UNCOMMENT THIS IF YOU ARE NOT DOING THE DEBUG BELOW
model.invLat{userId} = model.Lat{userId}\eye(size(model.Lat{userId},1));  
model.invLatT{userId} = model.invLat{userId}';
model.logDetAt{userId} = 2*(sum(log(diag(model.Lat{userId})))); % log |At|
model.P1{userId} = model.invLat{userId} * model.invLm{userId}; % M x M

% First multiply the two last factors; so, the large N is only involved
% once in the calculations (P1: MxM, Psi1':MxN, Y: NxD)
model.P{userId} = model.P1{userId} * (model.Psi1{userId}' * model.m{userId});
% Needed for both, the bound's and the derivs. calculations.
model.TrPP(userId) = sum(sum(model.P{userId}.* model.P{userId}));
%%% Precomputations for the derivatives (of the likelihood term) of the bound %%%

%model.B = model.invLmT * model.invLatT * model.P; %next line is better
model.B{userId} = model.P1{userId}' * model.P{userId};
model.invK_uu{userId} = model.invLmT{userId} * model.invLm{userId};
if model.onebeta
   Tb{userId} =  1/model.kern.beta * (model.P1{userId}' * model.P1{userId});
else
   Tb{userId} = (1/model.kern.beta(userId)) * (model.P1{userId}' * model.P1{userId});
end
Tb{userId} = Tb{userId} + (model.B{userId} * model.B{userId}');
model.T1{userId} = model.invK_uu{userId} - Tb{userId};


end
                            

%try %%%% DEBUG1
% model.Lat = jitChol(model.At)';
%catch e %%% DEBUG
%    model_At = model.At;
%    save('modellAt', 'model_At');
%    e.throw %%% DEBUG
%end %%%% DEBUG


% DEBUG2
% warning('') %%% DEBUG
% model.Lat = jitChol(model.At)';
% tmpWarn = lastwarn; %%% DEBUG
% if length(tmpWarn) > 41 && strcmp(tmpWarn(1:42), 'Matrix is not positive definite in jitChol') %%% DEBUG
%     % do stuff...
%     %%%model.id
%     warning('') % This is a good place to put a stop for the debugger...
% end %%% DEBUG





