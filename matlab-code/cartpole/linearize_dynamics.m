function [F0, JFX, JFu] = linearize_dynamics(X, u, cartpole)
    persistent F_Jac;   
    if isempty(F_Jac)
        F_Jac = casadi.Function.load('casadi_generated_functions/F_Jac.func');
    end
    params = [cartpole.m; cartpole.mc; cartpole.r; cartpole.g];
    [JFX, JFu] = F_Jac(X, u, params);
    JFX = full(JFX);
    JFu = full(JFu);
    F0 = dynamics(X, u, cartpole);
end
