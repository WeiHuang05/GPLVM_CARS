function model = vargplvmExpandParam(model, params)


% variational parameters (means and covariances), original ones
startVal = 1;
endVal = model.vardist.nParams*2;
vardistParams = params(startVal:endVal);

if ~isempty(model.vardist.transforms)
  for i = 1:length(model.vardist.transforms)
    index = model.vardist.transforms(i).index;
    fhandle = str2func([model.vardist.transforms(i).type 'Transform']);
    vardistParams(index) = fhandle(vardistParams(index), 'atox');
  end
end

means = vardistParams(1:model.vardist.nParams);
covs = vardistParams(model.vardist.nParams+1:end);

model.vardist.means = reshape(means, model.vardist.nParams, 1);
model.vardist.covars = reshape(covs, model.vardist.nParams, 1);


% Parameters include inducing variables.
startVal = endVal+1;
endVal = endVal + model.q*model.k;
model.X_u = reshape(params(startVal:endVal),model.k,model.q);


% kernel hyperparameters 
startVal = endVal+1; 
endVal = endVal + model.kern.nParams;
kernParams = params(startVal:endVal);

if ~isempty(model.kern.transforms)
  for i = 1:length(model.kern.transforms)
    index = model.kern.transforms(i).index;
    fhandle = str2func([model.kern.transforms(i).type 'Transform']);
    if isfield(model.kern.transforms(i),'transformsettings') && ~isempty(model.kern.transforms(i).transformsettings)    
      kernParams(index) = fhandle(kernParams(index), 'atox', model.kern.transforms(i).transformsettings);    
    else
      kernParams(index) = fhandle(kernParams(index), 'atox');    
    end
  end
end

if model.onevariance
   model.kern.variance = kernParams(1);
   startInverse = 1;
else
   model.kern.variance(1:model.d) = kernParams(1:model.d);
   startInverse = model.d;
end

if model.qinverse
   model.kern.inverseWidth(1:model.q) = kernParams(startInverse+1:startInverse+model.q);
   startbeta = startInverse+model.q;
else
   model.kern.inverseWidth(1:model.d) = kernParams(startInverse+1:startInverse+model.d);
   startbeta = startInverse+model.d;
end

if model.onebeta
   model.kern.beta = kernParams(end);
else
   model.kern.beta(1:model.d) = kernParams(startbeta+1:end);
end


model.nParams = endVal;
% Update statistics
model = vargplvmUpdateStats(model);

