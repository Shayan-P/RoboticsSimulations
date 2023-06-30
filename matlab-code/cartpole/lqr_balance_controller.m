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
    [F0, A, B] = linearize_dynamics(X0, 0, cartpole);
    Q = diag([10, 3, 3, 3]);
    R = 1;

    K = lqr(A, B, Q, R);

    u_cor = -K * X;
end
