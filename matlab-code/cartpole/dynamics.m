function dX = dynamics(X, u, cartpole)
    import casadi.*;

    persistent funcHandle;   
    if isempty(funcHandle)
        funcHandle = casadi.Function.load('casadi_generated_functions/dynamics.func');
    end
    params = [cartpole.m; cartpole.mc; cartpole.r; cartpole.g];
    dX = funcHandle(X, u, params);
end
