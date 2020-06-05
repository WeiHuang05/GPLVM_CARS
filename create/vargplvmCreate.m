function model = vargplvmCreate(train, test, options)

% data analysis
model.train = train;
model.test = test;
allT = [train; test];
model.num_users = max(allT(:,1));
model.num_items = max(allT(:,2));
model.num_context = max(allT(:,3:end-1));

% dimension setting
model.dim_item = 5;
model.dim_context = 5*ones(1,size(model.num_context,2));
model.d = model.num_users;
model.q = model.dim_item+sum(model.dim_context);
model.k = options.numActive;

% id setting
model.fiq(1:model.dim_item) = ones(1,model.dim_item);
qstart = model.dim_item;
for i = 1 : numel(model.num_context)
    model.fiq(qstart+1:qstart+model.dim_context(i)) = (i+1)*ones(1,model.dim_context(i));
    qstart = qstart + model.dim_context(i);
end

% environment setting
model.type = 'vargplvm';
model.optimiser = options.optimiser;
model.KLweight = 0.5;
model.learnBeta = 1;
model.date = date;
model.onebeta = options.onebeta; 
model.qinverse = options.qinverse;
model.onevariance = options.onevariance;

% loop calculation
min = 2000;
model.scale = ones(1, model.d);
for i = 1:model.num_users
    Tr = train(train(:,1) == i, :);
    Y{i} = Tr(:,end);
    diy = size(Y{i},1);
    model.numData(i) = diy;
    if diy<min
        min = diy;
    end
    model.bias{i} = mean(Y{i});
    model.m{i} = Y{i}-model.bias{i};
    if model.scale(i)
       model.m{i} = model.m{i}/model.scale(i);
    end
    model.TrYY(i) = sum(model.m{i}.*model.m{i});
end
model.N = min;

model.kern = kernCreate(model);
model.vardist = vardistCreate(model); 
model.X_u = induceCreate(model);
  
end

