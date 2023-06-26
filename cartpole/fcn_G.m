function [G] = fcn_G(q,p)

G = zeros(2,1, class(q));   %% Changed this line to support symbolic as well

  G(1,1)=0;
  G(2,1)=-p(4)*p(1)*p(3)*sin(q(2));

 