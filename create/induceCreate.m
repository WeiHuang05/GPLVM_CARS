function X_u = induceCreate(model)
% Sub-sample inducing variables.

if model.k <= model.N
   X_u = ppcaEmbed(model);             
else
 % Make it work even if k>N
   warning('k > N !')
   %rand(size(model.vardist.means(1:dif,:))).*sqrt(model.vardist.covars(1:dif,:));  % Sampling from a Gaussian.
   X_u=zeros(model.k, model.q);
   for i=1:model.k
       for j = 1:model.q
           ind=randperm(size(model.vardist.means,1));
           ind=ind(1);
           X_u(i,j) = model.vardist.means(ind,:);
       end
   end
           
end

%- Check if some inducing points are too close
res = util_checkCloseMatrixElements(X_u);
if ~isempty(res)
    warning('The following pairs of inducing points are too close!')
    for ii = 1:size(res,1)%length(res)
        fprintf('%s\n', num2str(res(ii,:)));
    end
end


