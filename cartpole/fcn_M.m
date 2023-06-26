function [M] = fcn_M(q,p)

M = zeros(2,2, class(q));  %% Changed this line to support symbolic as well

  M(1,1)= p(1) + p(2);
  M(1,2)=p(1)*p(3)*cos(q(2));
  M(2,1)=p(1)*p(3)*cos(q(2));
  M(2,2)=p(1)*p(3)^2;

 