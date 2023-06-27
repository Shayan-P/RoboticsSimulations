function u_cor = lqr_balance_controller(t, X, cartpole)
    import casadi.*

    persistent F_Jac;   
    if isempty(F_Jac)
        F_Jac = casadi.Function.load('casadi_generated_functions/F_Jac.func');
    end

    theta = mod(X(2), 2 * pi);
    if theta > pi
        theta = theta - 2 * pi;
    end
    X = X(:);
    X(2) = theta;
    %% normalize theta
    % normalize around a close balance:
    X0 = [X(1); 0; 0; 0];

    params = [cartpole.m; cartpole.mc; cartpole.r; cartpole.g];
    [A, B] = F_Jac(X0, 0, params);
    A = full(A);
    B = full(B);

    Q = diag([10, 3, 3, 3]);
    R = 1;

    K = lqr(A, B, Q, R);

    u_cor = -K * X;
end
