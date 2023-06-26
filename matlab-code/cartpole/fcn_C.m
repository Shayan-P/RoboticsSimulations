function [C] = fcn_C(q,dq,p)

C = zeros(2,2, class(q));   %% Changed this line to support symbolic as well

  C(1,1)=0;
  C(1,2)=-dq(2)*p(1)*p(3)*sin(q(2));
  C(2,1)=0;
  C(2,2)=0;

 