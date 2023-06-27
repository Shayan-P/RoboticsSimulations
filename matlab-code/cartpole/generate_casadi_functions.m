addpath(genpath('../casadi'))

import casadi.*

m = MX.sym('m');
mc = MX.sym('mc');
r = MX.sym('r');
g = MX.sym('g');
params = [m; mc; r; g];

p = MX.sym('p');
dp = MX.sym('dp');
theta = MX.sym('theta');
dtheta = MX.sym('dtheta');
u = MX.sym('u'); % control signal. force applied to the cart

PE = m * g * r * cos(theta);
V = [dp; 0.0] + r * dtheta * [cos(theta); -sin(theta)];
KE = 0.5 * (m * sum(V .* V) + mc * dp * dp);

q = [p; theta];
dq = [dp; dtheta];
X = [q; dq];
[M, C, G, B] = std_dynamics(KE, PE, q, dq, q);

ddq = pinv(M) * ([u; 0] - C * dq - G);
dX = [dq; ddq];

fcn_M = Function('M', {X, params}, {M}, {'X', 'p'}, {'M'});
fcn_C = Function('C', {X, params}, {C}, {'X', 'p'}, {'C'});
fcn_G = Function('G', {X, params}, {G}, {'X', 'p'}, {'G'});
fcn_dynamics = Function('dynamics', {X, u, params}, {dX}, {'X', 'u', 'p'}, {'dX'});

save(fcn_M, 'casadi_generated_functions/M.func')
save(fcn_C, 'casadi_generated_functions/C.func')
save(fcn_G, 'casadi_generated_functions/G.func')
save(fcn_dynamics, 'casadi_generated_functions/dynamics.func')

% tau = Mddq + Cdq + G
% ddq_X_u = @(X, u_var) pinv(M(X(2))) * ([u_var; 0.0] - C(X(4), X(2)) * reshape(X(3:4), (2, 1)) - G(X(2)))
% ddq = simplify(pinv(M) * ([u; 0.0] - C * dq - G))

%% aux functions
% author: Ben Morris and Eric Westervelt
function [D,C,G,B] = std_dynamics(KE,PE,x,dx, xrel) % all variables must be symbolic when passed, symbolic variables are returned
    import casadi.*;

	G=jacobian(PE,x).';

	tem=jacobian(KE,dx).'; % tem=simple(jacobian(KE,dx).'); 
	D=jacobian(tem,dx); %simple(jacobian(tem,dx));
    
	n=max(size(x));
    C = MX.zeros(n, n);
	for k=1:n
		for j=1:n
			C(k,j)=0;
			for i=1:n
				C(k,j)=C(k,j)+(1/2)*(jacobian(D(k,j),x(i)) + jacobian(D(k,i),x(j)) - jacobian(D(i,j),x(k)))*dx(i);
			end
		end
	end    
    B = jacobian(xrel,x)' ;
end
