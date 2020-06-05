function kern = kernCreate(model)

kern.transforms = [];

% variance
kern.variance = 1*ones(1, model.d);
num_varian = model.d;
if model.onevariance
    kern.variance = 1; 
    num_varian = 1;
end

% inversewidth
kern.inverseWidth = 0.999*ones(1, model.d);
num_inverse = model.d;
if model.qinverse
    kern.inverseWidth = 1*ones(1, model.q); 
    num_inverse = model.q;
end

% beta
kern.beta = 10*ones(1, model.d);
num_beta = model.d;
if model.onebeta
    kern.beta = 100;
    num_beta = 1;
end

kern.nParams = num_varian + num_inverse + num_beta;

kern.transforms(1).index = [1:kern.nParams];
kern.transforms(1).type = 'exp';


