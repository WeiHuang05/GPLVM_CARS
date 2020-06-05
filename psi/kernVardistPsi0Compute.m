function Psi0 = kernVardistPsi0Compute(model, kern, i)


if model.onevariance
   Psi0 = model.numData(i)*kern.variance;
else
   Psi0 = model.numData(i)*kern.variance(i);
end
  

