function SNR = vargplvmShowSNR(model, displ)

% VARGPLVMSHOWSNR A simple wrapper to display the Signal to Noise Ratio of an optimised varglpvm model.
% VARGPLVM

if nargin < 2
    displ = true;
end

%%


if isfield(model, 'mOrig') && ~isempty(model.mOrig)
    varY = var(model.mOrig(:));
else
       
    for i = 1:model.d
       varY1(i) = var(model.m{i});  
       if model.onebeta
           beta1(i) = model.kern.beta;
       else
           beta1(i) = model.kern.beta(i);
       end
    end
   
end
varY = mean(varY1);
beta = mean(beta1);
SNR = varY * beta;
if displ
    fprintf('     %f  (varY=%f, 1/beta=%f)\n',  SNR, varY, 1/beta)
end
%}

