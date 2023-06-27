function u_cor = lqr_balance_controller(t, X, cartpole)
    theta = mod(X(2), 2 * pi);
    if theta > pi
        theta = theta - 2 * pi;
    end
    X = X(:);
    X(2) = theta;
    %% normalize theta
    % normalize around a close balance:
    X0 = [X(1); 0; 0; 0];

    % dX = subs(jacobian(F_, X_), X_, X0, u_, 0) * X + subs(jacobian(F_, u_), X_, X0, u_, 0) * u

    X_ = sym('X', [4, 1]);
    u_ = sym('u');
    F_ = dynamics(X_, u_, cartpole);
    Fx_ = jacobian(F_, X_);
    Fu_ = jacobian(F_, u_);

    A = double(subs(subs(Fx_, X_, X0), u_, 0));
    B = double(subs(subs(Fu_, X_, X0), u_, 0));

    Q = diag([10, 3, 3, 3]);
    R = 1;

    K = lqr(A, B, Q, R);

    u_cor = -K * X + randn() * 0.01;
end
