function vardist = vardistCreate(model)

vardist.latentDimension = model.q; 
vardist.nParams = model.num_items*model.dim_item+sum(model.num_context.*model.dim_context);
%vardist.nParams = model.N*model.q;

vardist.type = 'vargplvm';

vardist.transforms(1).index = [vardist.nParams+1:2*vardist.nParams];
vardist.transforms(1).type = 'exp';

vardist.means = 0.1+0.1*randn(vardist.nParams,1);
%vardist.means = 0.1+0.1*randn(model.N, model.q);

vardist.covars = 0.01+0.001*randn(vardist.nParams,1);




