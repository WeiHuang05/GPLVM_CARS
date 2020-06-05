function ll = vargplvmLogLikelihood(model)

% VARGPLVMLOGLIKELIHOOD Log-likelihood for a variational GP-LVM.
% FORMAT
% DESC returns the log likelihood for a given GP-LVM model.
% ARG model : the model for which the log likelihood is to be
% computed. The model contains the data for which the likelihood is
% being computed in the 'y' component of the structure.
% RETURN ll : the log likelihood of the data given the model.
%
% COPYRIGHT : Michalis K. Titsias, 2009-2011
% COPYRIGHT : Neil D. Lawrence, 2009-2011
% COPYRIGHT : Andreas Damianou, 2010-2011


% VARGPLVM

% Note: The 'onlyKL' and 'onlyLikelihood' fields can be set by external
% wrappers and cause the function to only calculate the likelihood or the
% KL part of the variational bound.

% Likelihood term
ll = 0;
if ~(isfield(model, 'onlyKL') && model.onlyKL)
        for i = 1:model.d
            if model.onebeta
                    part1 = - 0.5*((-(model.numData(i)-model.k)*log(model.kern.beta) ...
                    + model.logDetAt{i}) ...
                    - (model.TrPP(i) ...
                    - model.TrYY(i))*model.kern.beta);
                
                    part2 = -0.5*model.kern.beta*model.Psi0(i); 
                    part3 = 0.5*model.kern.beta*model.TrC(i);
            else
                    part1 = - 0.5*((-(model.numData(i)-model.k)*log(model.kern.beta(i)) ...
                    + model.logDetAt{i}) ...
                    - (model.TrPP(i) ...
                    - model.TrYY(i))*model.kern.beta(i));
                
                    part2 = -0.5*model.kern.beta(i)*model.Psi0(i); 
                    part3 = 0.5*model.kern.beta(i)*model.TrC(i);
            end
            part4 = -model.numData(i)/2*log(2*pi);
%======================================================            
          %  part1 = 0;
          %  part2 = 0;
          %  part3 = 0;
          %  part4 = 0;
%======================================================
            ll = ll + part1+part2+part3+part4;
        end
        %end
else
    ll=0;
end


% KL divergence term
if ~(isfield(model, 'onlyLikelihood') && model.onlyLikelihood)
    if isfield(model, 'dynamics') && ~isempty(model.dynamics)
        % A dynamics model is being used. (The following is actually
        % the -KL term)
        KLdiv = modelVarPriorBound(model);
        KLdiv = KLdiv + 0.5*model.q*model.N; %%% The constant term!!
    else
        varmeans = sum(model.vardist.means.*model.vardist.means);
        varcovs = sum(model.vardist.covars - log(model.vardist.covars));
        %KLdiv = -0.5*(varmeans + varcovs) + 0.5*model.q*model.N;
        KLdiv = -0.5*(varmeans + varcovs) + 0.5*size(model.vardist.means,1);
    end
else
    KLdiv=0;
end

% ll = ll + KLdiv);

% Obtain the final value of the bound by adding the likelihood
% and the KL term (possibly weighted, although this is not tested and
% weights should better be both 0.5). But this trick is applied only in the
% normal optimisation loop, not when the var. distr. is optimised with SNR
% fixed.
%if model.initVardist
%    model.KLweight = 0.5;
%end


fw = model.KLweight;
%=====================================================
%fw = 0;
ll = 2*((1 - fw)*ll + fw * KLdiv);
%=====================================================

% If there's a prior on some parameters, add the contribution
%ll = ll + vargplvmParamPriorLogProb(model);

%--- Trick from PILCO v0.9 (Deisenroth et al.) to prevent very low SNR
% See vargplvmGradient for info about the params.
if isfield(model, 'SNRpenalty') && ~isempty(model.SNRpenalty) && model.SNRpenalty.flag
    p   = model.SNRpenalty.p;  % Default: 15
    snr = model.SNRpenalty.snr; % Default: 1000
    if isfield(model, 'mOrig')
        lsf = log(std(model.mOrig(:)));
    else
        lsf = log(std(model.m(:)));
    end
    lsn = log(sqrt(1/model.beta));
    % Original (PILCO) implementation didn't have 2* factor. But there they
    % apply the penalty also to the white term. Not 100% sure that the 2*
    % factor is the correct, but seems to work for the gradients...
    ll = ll - 2*sum(((lsf - lsn)/log(snr)).^p);
end
