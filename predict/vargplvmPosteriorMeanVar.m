function [mu, varsigma] = vargplvmPosteriorMeanVar(userId, model, vardistX)

Ainv = model.P1{userId}' * model.P1{userId}; % size: NxN

if ~isfield(model,'alpha')
    model.alpha = Ainv*model.Psi1{userId}'*model.m{userId}; % size: 1xD
end
Psi1_star = rbfard2VardistPsi1Compute(userId, model, model.kern, vardistX, model.X_u);

% mean prediction 
mu = Psi1_star*model.alpha; % size: 1xD

%{
if nargout > 1
   % 
   % precomputations
   vard = vardistCreate(zeros(1,model.q), model.q, 'gaussian');
   Kinvk = (model.invK_uu - (1/model.beta)*Ainv);
   %
   for i=1:size(vardistX.means,1)
      %
      vard.means = vardistX.means(i,:);
      vard.covars = vardistX.covars(i,:);
      % compute psi0 term
      Psi0_star = kernVardistPsi0Compute(model.kern, vard);
      % compute psi2 term
      Psi2_star = kernVardistPsi2Compute(model.kern, vard, model.X_u);
    
      vars = Psi0_star - sum(sum(Kinvk.*Psi2_star));
      
      for j=1:model.d
         %[model.alpha(:,j)'*(Psi2_star*model.alpha(:,j)), mu(i,j)^2]
         varsigma(i,j) = model.alpha(:,j)'*(Psi2_star*model.alpha(:,j)) - mu(i,j)^2;  
      end
      varsigma(i,:) = varsigma(i,:) + vars; 
      %
   end
   % 
   if isfield(model, 'beta')
      varsigma = varsigma + (1/model.beta);
   end
   %
end
%}
      
% Rescale the mean
mu = mu.*repmat(model.scale(userId), size(vardistX.means,1), 1);

% Add the bias back in
mu = mu + repmat(model.bias{userId}, size(vardistX.means,1), 1);

% rescale the variances
if nargout > 1
    %varsigma = varsigma.*repmat(model.scale.*model.scale, size(vardistX.means,1), 1);
    varsigma = 1;
end
  