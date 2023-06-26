function dX = dynamics(X, u, cartpole)
    [p, theta, dp, dtheta] = deal(X(1), X(2), X(3), X(4));
    q = [p; theta];
    dq = [dp; dtheta];
    p = [cartpole.m; cartpole.mc; cartpole.r; cartpole.g];
    M = fcn_M(q, p);
    C = fcn_C(q, dq, p);
    G = fcn_G(q, p);
    ddq = pinv(M) * ([u; 0] - C * dq - G);
    dX = [dq; ddq];
end