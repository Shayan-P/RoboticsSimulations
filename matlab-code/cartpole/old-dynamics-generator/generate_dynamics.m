m = sym('m')
mc = sym('mc')
r = sym('r')
g = sym('g')
params = [m;mc;r;g]
m_list_params = gen_m_lists(params,'p');

p = sym('p')
dp = sym('dp')
theta = sym('theta')
dtheta = sym('dtheta')
u = sym('u') % control signal. force applied to the bottom

PE = m * g * r * cos(theta)
V = [dp; 0.0] + r * dtheta * [cos(theta); -sin(theta)]
KE = 0.5 * (m * sum(V .* V) + mc * dp * dp)

q = [p; theta]
dq = [dp; dtheta]
[M, C, G, B] = std_dynamics(KE, PE, q, dq, q)

m_list_q = gen_m_lists(q,'q');
m_list_dq = gen_m_lists(dq,'dq');

write_fcn_m('fcn_M.m',{'q', 'p'},[m_list_q;m_list_params],{M,'M'});
write_fcn_m('fcn_C.m',{'q','dq', 'p'},[m_list_q; m_list_dq; m_list_params],{C,'C'});
write_fcn_m('fcn_G.m',{'q', 'p'}, [m_list_q; m_list_params], {G,'G'});

% tau = Mddq + Cdq + G
% ddq_X_u = @(X, u_var) pinv(M(X(2))) * ([u_var; 0.0] - C(X(4), X(2)) * reshape(X(3:4), (2, 1)) - G(X(2)))
% ddq = simplify(pinv(M) * ([u; 0.0] - C * dq - G))

%% aux functions
% author: Ben Morris and Eric Westervelt
function [D,C,G,B] = std_dynamics(KE,PE,x,dx, xrel) % all variables must be symbolic when passed, symbolic variables are returned
	G=jacobian(PE,x).';
	G=simple_elementwise(G); 

	tem=jacobian(KE,dx).'; % tem=simple(jacobian(KE,dx).'); 
	D=simple_elementwise(jacobian(tem,dx)); %simple(jacobian(tem,dx));
    
	syms C
	n=max(size(x));
	for k=1:n
		for j=1:n
			C(k,j)=0;
			for i=1:n
				C(k,j)=C(k,j)+(1/2)*(diff(D(k,j),x(i)) + diff(D(k,i),x(j)) - diff(D(i,j),x(k)))*dx(i);
			end
		end
	end
	C=simple_elementwise(C);%simple(C);
    
    B = jacobian(xrel,x)' ;
end

% Function to perform simple() one element at a time
function M = simple_elementwise(M)
    for i=1:size(M,1)
        for j=1:size(M,2)
            M(i,j) = simplify(M(i,j)) ;
        end
    end
end

function m_list = gen_m_lists(vec, str_prefix)
    % author: Eric Westervelt
    % generate list for gen_dyn_boom_leg
    m_list = {} ;
    for j=1:length(vec)
        m_list{j,1} = char(vec(j)) ;
        m_list{j,2} = [str_prefix '(' num2str(j) ')'] ;
    end
end
