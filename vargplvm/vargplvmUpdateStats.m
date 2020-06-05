function model = vargplvmUpdateStats(model)
% VARGPLVMUPDATESTATS Update stats for VARGPLVM model.
%
% COPYRIGHT : Michalis K. Titsias, 2009-2011
% COPYRIGHT : Neil D. Lawrence, 2009-2011
% COPYRIGHT : Andreas C. Damianou, 2010-2011

  
jitter = 1e-6;
%model.jitter = 1e-6;

%%% Precomputations for (the likelihood term of) the bound %%%
for i = 1 : model.d      
 %   model.K_uu{i} = kernCompute(model,i);
    model.K_uu{i} = rbfard2KernCompute(model, i);
  
    model.K_uu{i} = model.K_uu{i} ...
                         + sparseDiag(repmat(jitter, size(model.K_uu{i}, 1), 1));
 
    model.Psi0(i) = kernVardistPsi0Compute(model, model.kern,i);
    model.Psi1{i} = kernVardistPsi1Compute(model, model.kern,i);
    model.Psi2{i} = kernVardistPsi2Compute(model, model.kern,i);
    
    %model.Lm = chol(model.K_uu, 'lower'); 
    model.Lm{i} = jitChol(model.K_uu{i})';      % M x M: L_m (lower triangular)   ---- O(m^3)
    model.invLm{i} = model.Lm{i}\eye(model.k);  % M x M: L_m^{-1}                 ---- O(m^3)
    model.invLmT{i} = model.invLm{i}'; % L_m^{-T}
    model.C{i} = model.invLm{i} * model.Psi2{i} * model.invLmT{i};
    model.TrC(i) = sum(diag(model.C{i})); % Tr(C)   Tr(ABC)=Tr(BCA)
% Matrix At replaces the matrix A of the old implementation; At is more stable
% since it has a much smaller condition number than A=sigma^2 K_uu + Psi2
    if model.onebeta
       model.At{i} = 1/model.kern.beta * eye(size(model.C{i},1)) + model.C{i}; % At = beta^{-1} I + C
    else
        model.At{i} = (1/model.kern.beta(i)) * eye(size(model.C{i},1)) + model.C{i}; % At = beta^{-1} I + C
    end
    model.Lat{i} = jitChol(model.At{i})'; % UNCOMMENT THIS IF YOU ARE NOT DOING THE DEBUG BELOW
    
    model.invLat{i} = model.Lat{i}\eye(size(model.Lat{i},1));  
    model.invLatT{i} = model.invLat{i}';
    model.logDetAt{i} = 2*(sum(log(diag(model.Lat{i})))); % log |At|

    model.P1{i} = model.invLat{i} * model.invLm{i}; % M x M

% First multiply the two last factors; so, the large N is only involved
% once in the calculations (P1: MxM, Psi1':MxN, Y: NxD)
    model.P{i} = model.P1{i} * (model.Psi1{i}' * model.m{i});
% Needed for both, the bound's and the derivs. calculations.
    model.TrPP(i) = sum(sum(model.P{i}.* model.P{i}));
%%% Precomputations for the derivatives (of the likelihood term) of the bound %%%

%model.B = model.invLmT * model.invLatT * model.P; %next line is better
    model.B{i} = model.P1{i}' * model.P{i};
    model.invK_uu{i} = model.invLmT{i} * model.invLm{i};
    if model.onebeta
       Tb{i} =  1/model.kern.beta * (model.P1{i}' * model.P1{i});
    else
       Tb{i} = (1/model.kern.beta(i)) * (model.P1{i}' * model.P1{i});
    end
	Tb{i} = Tb{i} + (model.B{i} * model.B{i}');
    model.T1{i} = model.invK_uu{i} - Tb{i};


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





