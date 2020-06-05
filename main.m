
clear;
addpath('data');
addpath('opt');
addpath('psi');
addpath('kern');
addpath('tools');
addpath('create');
addpath('predict');
addpath('vargplvm');

% Fix seeds
rng('default');
rng(3);
randn('seed', 1e5);
rand('seed', 1e5);

% load data
for fn = 1:5

datafilename = ['food'  num2str(fn) ]; 
[train,test] = lvmLoadData(datafilename);

% Set up model
options.kern = 'rbf';
options.optimiser = 'sgd'; %options.optimiser = 'scg2';
options.numActive = 18;    % induce points
options.onebeta = 0;       % user centered beta
options.qinverse = 1;      % user centered inverse length
options.onevariance = 0;   % user centered variance

model = vargplvmCreate(train, test, options);

% Set up parameters
model.kern.variance = 1+0.1*rand(1,model.d);
model.kern.beta = 10+0.1*rand(1,model.d); 
model.vardist.means = 0.1+0.1*randn(model.num_items+sum(model.num_context),model.dim_item);
model.vardist.covars = 0.002*ones(model.vardist.nParams,1) + 0.0001*randn(model.vardist.nParams,1);

model = vargplvmParamInit(model); 

% Optimise the model.
iters = 20;
display = 1;

model = vargplvmOptimise(fn, model, display, iters);

% Predict
[evaluation,fpred] = rPredictAll(model,test);
evaluation = evaluation(1:2);
fprintf('mae \t rmse \n');
fprintf('%.4f\t%.4f\n', evaluation);

% Save 
filename = ['oup\model'  num2str(fn) '.mat'];
save(filename, 'model');

end

%% 
