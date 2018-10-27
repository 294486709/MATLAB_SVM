function sim = PolynominalKernel(x1, x2)
x1 = x1(:); x2 = x2(:);
sim = 0;
sim = (1*x1'*x2+1)^5;    
end