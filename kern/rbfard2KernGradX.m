function gX = rbfard2KernGradX(model, kern, X, X2, d)

% RBFARD2KERNGRADX Gradient of RBFARD2 kernel with respect to input locations.
% FORMAT
% DESC computes the gradident of the automatic relevance determination radial basis function
% kernel with respect to the input positions where both the row
% positions and column positions are provided separately.
% ARG kern : kernel structure for which gradients are being
% computed.
% ARG x1 : row locations against which gradients are being computed.
% ARG x2 : column locations against which gradients are being computed.
% RETURN g : the returned gradients. The gradients are returned in
% a matrix which is numData2 x numInputs x numData1. Where numData1 is
% the number of data points in X1, numData2 is the number of data
% points in X2 and numInputs is the number of input
% dimensions in X.
%
% SEEALSO rbfard2KernParamInit, kernGradX, rbfard2KernDiagGradX
%
% COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006
%
% COPYRIGHT : Michalis K. Titsias, 2009

% KERN


gX = zeros(size(X2, 1), size(X2, 2), size(X, 1));
for i = 1:size(X, 1)
  gX(:, :, i) = rbfard2KernGradXpoint(model, kern, X(i, :), X2, d);
end
  

function gX = rbfard2KernGradXpoint(model, kern, x, X2, d)


% RBFARD2KERNGRADXPOINT Gradient with respect to one point of x.
Q = size(x,2);
   
if model.qinverse
   invQ = sqrt(kern.inverseWidth);
else
   invQ = sqrt(kern.inverseWidth(d));
   invQ = repmat(invQ,[1,Q]);
end
   
scales = sparse(diag(invQ));
gX = zeros(size(X2));
n2 = dist2(X2*scales, x*scales);

if model.onevariance
    rbfPart = kern.variance*exp(-n2*0.5);
else
    rbfPart = kern.variance(d)*exp(-n2*0.5);
end

for i = 1:size(x, 2)
   if model.qinverse
       gX(:, i) = kern.inverseWidth(i)*(X2(:, i) - x(i)).*rbfPart;
   else
       gX(:, i) = kern.inverseWidth(d)*(X2(:, i) - x(i)).*rbfPart;
   end
end
