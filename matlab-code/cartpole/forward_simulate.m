function sim_trajectory = forward_simulate(controller, problem)
    ts = linspace(0, problem.T, 100);
    [tout, Xout] = ode45(@(t,X) dynamics(X, controller(t, X), problem.cartpole), ts, problem.X0);
    n = size(tout, 1);
    uout = [];
    for i=1:n
        uout = [uout; controller(tout(i), Xout(i, :)')]; 
    end
    sim_trajectory.t = tout;
    sim_trajectory.X = Xout;
    sim_trajectory.u = uout;
end
