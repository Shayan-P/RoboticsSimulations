addpath('/home/shayan/Desktop/mit/robotics-urop-biomimic-lab/casadi-3.6.3-linux64-matlab2018b');

import casadi.*

param.r = 0.2;
param.m = 1;
param.mc = 0.5;
param.g = 9.8;

T = 2; % control horizon [s]
N = 160; % Number of control intervals

% X0 = [0; 0.1; 0; 0];
% T = 10;
% controller = @(t, X) 0;
% test_controller(controller, X0, T, param)

T = 2;
N = 160;
DT = T/N;

init_X = [0;pi;0;0];
final_X = [1;0;0;0];

%% time evolution integration
X = MX.sym('X', 4, 1);
u = MX.sym('u');

%options.number_of_finite_elements = 1;
%options.simplify = true;
%options.tf = DT/options.number_of_finite_elements;

duration = MX.sym('duration')
intg = integrator('intg', 'rk', struct('x', X, 'ode', dXdt(X, u, param) * duration, 'p', [u, duration]), struct('tf', 1));
res = intg('x0',X,'p', [u, duration]);
xf = full(res.xf);
FT = Function('FT', {X, u, duration}, {xf});
F = Function('F', {X, u}, {FT(X, u, DT)});

%% correction

Xd = MX.sym('Xd', 4, 1);
% F(X, u, duration) = Xd -> find u
rf_param = struct('x', u, 'p', [X; Xd; duration], 'g', norm(Xd - FT(X, u, duration)));
rf = rootfinder('rf', 'newton', rf_param, struct('error_on_fail', false));
u0 = MX.sym('u0')
solver = rf('x0', u0, 'p', [X; Xd; duration]);
corrector = Function('corrector', {u0, X, Xd, duration}, {full(solver.x)});

%% constructing optimization problem

opti = Opti();
Xs = opti.variable(N+1, 4);
us = opti.variable(N, 1);
opti.subject_to(Xs(1,:)' == init_X); % initial values
objective = 0;

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


tout = linspace(0, T, N+1)';
Xout = opti.value(Xs);
uout = [0; opti.value(us)];

animate_func = @(tout, Xout) animate(tout, Xout, param);

% show the result of optimizer
animate(tout, Xout, param);

% simulate with forward dynamics
controller = @(t, X) uout(max(find(tout <= t))); % don't care about X. its supposed to be calculated right
pd_controller_gen = @(p, d) (@(t, X) pd_controller(tout, Xout, uout, t, X, p, d));
controller = pd_controller_gen(1, 0.1);
% controller = @(t, X) main_controller(tout, Xout, uout, t, X, corrector);
test_controller_func = @(controller) test_controller(controller, init_X, T, param);
[tout_real, Xout_real, uout_real] = test_controller_func(controller); 

% stats
clf;
subplot(2, 1, 1)
plot(tout, Xout, tout, uout);
legend({'p', 'theta', 'dp', 'dtheta'});
title('optimization')

subplot(2, 1, 2)
plot(tout_real, Xout_real, tout_real, uout_real);
legend({'p', 'theta', 'dp', 'dtheta'});
title('real')

%% another controller that uses root finding
function u = main_controller(tout, Xout, uout, t, X, corrector)
    index = max(find(tout <= t));
    if index == size(tout, 1)
        u = uout(index); % end case
    else
        u = uout(index + 1);
        td = tout(index+1, :);
        Xd = Xout(index+1, :)';
        u = full(corrector(u, X, Xd, td-t));
    end
end
%% simple pd controller
function u = pd_controller(tout, Xout, uout, t, X, pcof, dcof)
    index = max(find(tout <= t));
    ud = uout(index, :);
    Xd = Xout(index, :)';
    qd = Xd(1:2);
    dqd = Xd(3:4);
    q = X(1:2);
    dq = X(3:4);
    e = qd - q; 
    de = dqd - dq;
    % u = ud + pcof * pinv(de/du) * e + dcof * pinv(dde/du) * de ??!
    % u = ud + pcof * sum(e) + dcof * sum(de);
    u = ud + pcof * e(1) + dcof * de(1);
end

%% dynamics
function dX = dXdt(X, u, param)
    [p, theta, dp, dtheta] = deal(X(1), X(2), X(3), X(4));
    q = [p; theta];
    dq = [dp; dtheta];
    p = [param.m; param.mc; param.r; param.g];
    M = fcn_M(q, p);
    C = fcn_C(q, dq, p);
    G = fcn_G(q, p);
    ddq = pinv(M) * ([u; 0] - C * dq - G);
    dX = [dq; ddq];
end

%% test controller
function [tout, Xout, uout] = test_controller(controller, X0, T, param)
    tstart = 0;
    tfinal = T;
    Div = 5;
    tend = tfinal / Div;

    tout = [];
    Xout = [];
    uout = [];
    for ii = 1:Div
        [t_,X_] = ode45(@(t,X)dXdt(X, controller(t, X), param),[tstart,tend],X0);    
        tstart = tend;
        tend = tend + tfinal/Div;
        X0 = X_(end,:)';
        tout = [tout;t_];
        Xout = [Xout;X_];   
        size_t_ = size(t_);
        n = size_t_(1);
        uout = [uout; reshape((arrayfun(@(i) controller(t_(i, :), X_(i, :)'), 1:n)), size_t_)];
    end
    
    animate(tout, Xout, param);

end

%% animation
function animate(t,X,p)
    clf;
    hold on

    h_link = plot(0,0,'linewidth',2,'color','k');
    h_target = plot(0,0,'ro');

    len = length(t);
    t(end+1) = t(end);
    
    for ii = 1:len
        [posx, theta] = deal(X(ii,1), X(ii, 2));
        
        posd = [posx + p.r * sin(theta); p.r * cos(theta)];
        h_target.XData = posd(1);
        h_target.YData = posd(2);
        
        chain = [
            [posx; 0], [posd]
        ];
        h_link.XData = chain(1,:);
        h_link.YData = chain(2,:);
        
        limit = 0.8;
        mx = max([0.8; abs(posx); abs(posd)]);
        xlim(mx *[-1.2 1.2])
        ylim(mx *[-1.2 1.2])
        title(['t = ',num2str(t(ii),3),' s'])
        axis square
        pause(t(ii+1)-t(ii));
    end
end


function res = elif(a, b, condition)
    if condition
        res = a;
    else
        res = b;
    end
end

