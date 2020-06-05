function [fmean, fvar] = predict(userId, model, T)
%PREDICT predict(userId, model, testObservations)
% 
% Predict using the learned model.
% 
% INPUT
%   - userId : id of the user to make prediction
%   - model : the learned model
%   - testObservations : same format as model.T
%
% OUTPUT

rows = find(T(:,1) == userId);   % all observations for userId
M = numel(model.num_context);   % number of contexts

item_means  = model.vardist.means(1:model.dim_item*model.num_items);
item_covars = model.vardist.covars(1:model.dim_item*model.num_items);
X = [];
S = [];
for d1 = 1:model.dim_item
    X = [X, item_means(T(rows,2)+(d1-1)*model.num_items,:)];
    S = [S, item_covars(T(rows,2)+(d1-1)*model.num_items,:)];
end



ind = model.dim_item*model.num_items;
for k = 1:M 
    context_means{k}  = model.vardist.means(ind+1:ind+model.dim_context(k)*model.num_context(k));
    context_covars{k} = model.vardist.covars(ind+1:ind+model.dim_context(k)*model.num_context(k));
    for d2 = 1:model.dim_context(k)
        X = [X, context_means{k}(T(rows,k+2)+(d2-1)*model.num_context(k),:)]; 
        S = [S, context_covars{k}(T(rows,k+2)+(d2-1)*model.num_context(k),:)]; 
    end
    ind = ind+model.dim_context(k)*model.num_context(k);
end

vardistX.means = X;
vardistX.covars = S;

[fmean, fvar] =  vargplvmPosteriorMeanVar(userId, model, vardistX);

end


