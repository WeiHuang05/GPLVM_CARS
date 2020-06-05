function [x, options, flog, pointlog, scalelog] = sgd(fn, model,  x, options, gradf, varargin)


nparams = length(x);

opt.epochs = 150;
opt.momentum = 0.9;
opt.learnRate = 0.5000e-04;

for j = 1:opt.epochs
    sampleIds = randperm(model.num_users);
    
    for i=1:model.num_users
       userId = sampleIds(i);
       if ~any(model.train(:,1) == userId),    continue,   end;
       if (j==1)
          delta{userId} = [];
       end
       %model = vargplvmExpandParam(model, x);
       model = stochasticExpandParam(model, x, userId);
       g =  -vargplvmstochasticGradients(userId, model);
   
       if isempty(delta{userId})
           delta{userId} = zeros(1,nparams);
       end
       delta{userId} = opt.momentum * delta{userId} - opt.learnRate * g;
       xnew = x + delta{userId};
       x = xnew;
    end
    
    f = -vargplvmLogLikelihood(model);
    fprintf(1, 'Cycle %4d  Error %11.10f \n', j, f);

  
  model = vargplvmExpandParam(model, x);
  [evaluation,~] = rPredictAll(model,model.test);
  evaluation = evaluation(1:2);
  fprintf('mae \t rmse \n');
  fprintf('%.4f\t%.4f\n', evaluation);
  %model.kern.inverseWidth(1:5)
  
  xrow(j) = j;
  yrow1(j) = f;
  
  yrow2(j) = evaluation(1);
  yrow3(j) = evaluation(2);
  
end

filename = ['oup\file'    num2str(fn)  '.mat' ];
save(filename,'xrow','yrow1','yrow2','yrow3'); 




