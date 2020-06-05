function [params] = vargplvmExtractParam(model)

% vardist
VarParams = [model.vardist.means(:)' model.vardist.covars(:)'];
% Check if parameters are being optimised in a transformed space.
if ~isempty(model.vardist.transforms)
  for i = 1:length(model.vardist.transforms)
    index = model.vardist.transforms(i).index;
    fhandle = str2func([model.vardist.transforms(i).type 'Transform']);
    VarParams(index) = fhandle(VarParams(index), 'xtoa');
  end
end

params = VarParams;

% inducing point
params =  [params model.X_u(:)'];
 
% kern
KernParams = [model.kern.variance  model.kern.inverseWidth  model.kern.beta];

if ~isempty(model.kern.transforms)
  for i = 1:length(model.kern.transforms)
    index = model.kern.transforms(i).index;
    fhandle = str2func([model.kern.transforms(i).type 'Transform']);
    
    if isfield( model.kern.transforms(i),'transformsettings' ) && ~isempty(kern.transforms(i).transformsettings')
      KernParams(index) = fhandle(KernParams(index), 'xtoa', kern.transforms(i).transformsettings);
    else
      KernParams(index) = fhandle(KernParams(index), 'xtoa');
    end    
  end
end

params = [params KernParams];




