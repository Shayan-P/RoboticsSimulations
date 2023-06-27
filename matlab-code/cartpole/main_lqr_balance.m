% addpath(genpath('../MatlabProgressBar'))

problem.T = 6;
problem.N = 160;
problem.X0 = [-3; 0.1; 0; 0];

problem.cartpole.r = 0.2;
problem.cartpole.m = 1;
problem.cartpole.mc = 0.5;
problem.cartpole.g = 9.8;

my_controller = @(t, X) lqr_balance_controller(t, X, problem.cartpole);
sim_trajectory = forward_simulate(my_controller, problem);
figure(1);
animate_cartpole(sim_trajectory.t, sim_trajectory.X, problem.cartpole);