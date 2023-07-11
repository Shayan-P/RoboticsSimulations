function u = controller(trajectory, t, X, cartpole)
    Xd = interp1(trajectory.t, trajectory.X, t)';
    ud = interp1(trajectory.t, trajectory.u, t)';
    [F0, A, B] = linearize_dynamics(Xd, ud, cartpole);
    Q = diag([1, 1, 1, 1]);
    R = 1;
    try 
        K = lqr(A, B, Q, R);
        u = ud - K * (X-Xd);
    catch
        disp({'lqr failed at', t})
        u = ud;
    end
end
