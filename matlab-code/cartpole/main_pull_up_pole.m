addpath(genpath('../casadi'))

import casadi.*

problem.T = 2;
problem.N = 160;
problem.X0 = [0; pi; 0; 0];
problem.Xt = [1; 0; 0; 0];

problem.cartpole.r = 0.2;
problem.cartpole.m = 1;
problem.cartpole.mc = 0.5;
problem.cartpole.g = 9.8;

trajectory = find_trajectory(problem);

% show the result of optimizer
figure(1);
animate_cartpole(trajectory.t, trajectory.X, problem.cartpole);

% simulate with forward dynamics
% controller = @(t, X) uout(max(find(tout <= t))); % don't care about X. its supposed to be calculated right
my_controller = @(t, X) controller(trajectory, t, X);
sim_trajectory = forward_simulate(my_controller, problem);
figure(2);
animate_cartpole(sim_trajectory.t, sim_trajectory.X, problem.cartpole);

% stats
figure(3);

subplot(2, 2, 1)
plot(trajectory.t, trajectory.X);
legend({'p', 'theta', 'dp', 'dtheta'});
title('optimization X')

subplot(2, 2, 2)
plot(trajectory.t, trajectory.u)
legend({'u'})
title('optimization u')

subplot(2, 2, 3)
plot(sim_trajectory.t, sim_trajectory.X);
legend({'p', 'theta', 'dp', 'dtheta'});
title('simulation X')

subplot(2, 2, 4)
plot(sim_trajectory.t, sim_trajectory.u)
legend({'u'})
title('simulation u')


% how good followed trajectory
figure(4);

subplot(2, 1, 1)
plot(trajectory.t, trajectory.X(:, 1))
hold on;
plot(sim_trajectory.t, sim_trajectory.X(:, 1))
legend({'opt p', 'sim p'})

subplot(2, 1, 2)
plot(trajectory.t, trajectory.X(:, 2))
hold on;
plot(sim_trajectory.t, sim_trajectory.X(:, 2))
legend({'opt theta', 'sim theta'})