function model = vargplvmParamInit(model)

%{
load('data\comapremodel.mat');
model.vardist.means = mod.means;
model.vardist.covars = mod.covrs;
model.kern.variance = mod.varia*ones(1,model.d);
model.kern.inverseWidth = mod.insca(1)*ones(1,model.d);
model.kern.beta = mod.beta*ones(1,model.d);
model.X_u = mod.X_u;
%}

initParams = vargplvmExtractParam(model); 
model.numParams = length(initParams);
model = vargplvmExpandParam(model, initParams); 

