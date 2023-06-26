function trajectory = find_trajectory(problem)
    import casadi.*;

    T = problem.T;
    N = problem.N;
    DT = problem.T / problem.N;
    init_X = problem.X0;
    final_X = problem.Xt;
    cartpole = problem.cartpole;

    %% time evolution integration
    X = MX.sym('X', 4, 1);
    u = MX.sym('u');

    %options.number_of_finite_elements = 1;
    %options.simplify = true;
    %options.tf = DT/options.number_of_finite_elements;

    duration = MX.sym('duration');
    intg = integrator('intg', 'rk', struct('x', X, 'ode', dynamics(X, u, cartpole) * duration, 'p', [u, duration]), struct('tf', 1));
    res = intg('x0',X,'p', [u, duration]);
    xf = full(res.xf);
    FT = Function('FT', {X, u, duration}, {xf});
    F = Function('F', {X, u}, {FT(X, u, DT)});

    %% constructing optimization problem
    opti = Opti();
    Xs = opti.variable(N+1, 4);
    us = opti.variable(N, 1);
    opti.subject_to(Xs(1,:)' == init_X); % initial values

    for i=1:N
        opti.subject_to(F(Xs(i, :)', us(i, :)') == Xs(i+1, :)');
    end

    opti.subject_to(Xs(N+1, :)' == final_X)

    % you may have to remove this constraints for different tasks:
    % opti.subject_to(-pi/6 <= Xs <= pi/6);
    MXU = opti.variable();
    opti.subject_to(-MXU <= us <= MXU);
    opti.minimize(MXU);
    % 
    opti.solver('ipopt');
    opti.solve();


    trajectory.t = linspace(0, T, N+1)';
    trajectory.X = opti.value(Xs);
    trajectory.u = [0; opti.value(us)];
end
