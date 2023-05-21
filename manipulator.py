import sympy as smp
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from visualization import Animator, AnimatableKSegmentData, AnimatableLinePlot
from math import *


L, m, t, g = smp.symbols('L m t g')
theta1, theta2 = smp.Function("theta1")(t), smp.Function("theta2")(t)

dtheta1, dtheta2 = smp.diff(theta1, t), smp.diff(theta2, t)
ddtheta1, ddtheta2 = smp.diff(dtheta1, t), smp.diff(dtheta2, t)

p1 = np.array([
    L * smp.cos(theta1),
    L * smp.sin(theta1)])

p2 = p1 + np.array([
    L * smp.cos(theta1 + theta2),
    L * smp.sin(theta1 + theta2)
])

v1 = np.array([
    smp.diff(p1[0], t),
    smp.diff(p1[1], t)
])

v2 = np.array([
    smp.diff(p2[0], t),
    smp.diff(p2[1], t)
])

K = m * ((v1**2).sum() + (v2**2).sum()) / 2
P = m *g * (p1[1] + p2[1])

Lagrangian = K - P


tau1 = smp.diff(
    smp.diff(Lagrangian, dtheta1),t
) - smp.diff(Lagrangian, theta1)

tau2 = smp.diff(
    smp.diff(Lagrangian, dtheta2),t
) - smp.diff(Lagrangian, theta2)


_theta1 = smp.symbols('theta1')
_theta2 = smp.symbols('theta2')
_dtheta1 = smp.symbols('dtheta1')
_dtheta2 = smp.symbols('dtheta2')
_ddtheta1 = smp.symbols('ddtheta1')
_ddtheta2 = smp.symbols('ddtheta2')

tau1 = tau1.subs({ddtheta1: _ddtheta1, ddtheta2: _ddtheta2}).subs({dtheta1: _dtheta1, dtheta2: _dtheta2}).subs({theta1: _theta1, theta2: _theta2})
tau2 = tau2.subs({ddtheta1: _ddtheta1, ddtheta2: _ddtheta2}).subs({dtheta1: _dtheta1, dtheta2: _dtheta2}).subs({theta1: _theta1, theta2: _theta2})


params = {
    g: 9.8,
    m: 1,
    L: 0.3
}

forward_dynamics = smp.solve([tau1.subs(params), tau2.subs(params)],
                             [_ddtheta1, _ddtheta2])
forward_dynamics = {theta: exp.simplify() for theta, exp in forward_dynamics.items()}

# q : theta1, theta2
X = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64) # state: q, dq


def forw_dynamics_theta1(theta1, theta2, dtheta1, dtheta2):
    return 1.5*(1.33333333333333*dtheta1**2*sin(theta2) + 0.666666666666666*dtheta1**2*sin(2*theta2) + 2.66666666666667*dtheta1*dtheta2*sin(theta2) + 1.33333333333333*dtheta2**2*sin(theta2) - 65.3333333333333*cos(theta1) + 21.7777777777778*cos(theta1 + 2*theta2))/(3 - cos(2*theta2))


def forw_dynamics_theta2(theta1, theta2, dtheta1, dtheta2):
    return 1.5*(-2.0*dtheta1**2*sin(theta2) - 0.666666666666667*dtheta1**2*sin(2*theta2) - 1.33333333333333*dtheta1*dtheta2*sin(theta2) - 0.666666666666667*dtheta1*dtheta2*sin(2*theta2) - 0.666666666666667*dtheta2**2*sin(theta2) - 0.333333333333333*dtheta2**2*sin(2*theta2) + 32.6666666666667*cos(theta1) + 21.7777777777778*cos(theta1 - theta2) - 21.7777777777778*cos(theta1 + theta2) - 10.8888888888889*cos(theta1 + 2*theta2))/(sin(theta2)**2 + 1)


def dstate_dt(t, X):
    return np.array([
            *X[2:],
            forw_dynamics_theta1(theta1=X[0], theta2=X[1], dtheta1=X[2], dtheta2=X[3]),
            forw_dynamics_theta2(theta1=X[0], theta2=X[1], dtheta1=X[2], dtheta2=X[3])
    ])

dt = 0.001
cur_t = 0
T = 10
points = []
l1_length = []
l2_length = []
ts = []

p1_str = [str(elem) for elem in p1]
p2_str = [str(elem) for elem in p2]

while cur_t < T:
    theta1 = X[0]
    theta2 = X[1]
    p1 = params[L] * np.array([np.cos(theta1), np.sin(theta1)])
    p2 = p1 + params[L] * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2)])
    points.append([
        [0, 0],
        p1, p2
    ])
    l1_length.append(np.linalg.norm(p1))
    l2_length.append(np.linalg.norm(p2-p1))
    ts.append(cur_t)
    cur_t += dt
    X = integrate.odeint(dstate_dt, X, [0, dt], tfirst=True)[1]

fig, (main_ax, l1_ax, l2_ax) = plt.subplots(3, 1)

segments = AnimatableKSegmentData(main_ax, points_data=points, name="two link robot", fix_scale=True)
l1_log = AnimatableLinePlot(l1_ax, x_data=ts, y_data=l1_length, name="l1", fix_scale=False)
l2_log = AnimatableLinePlot(l2_ax, x_data=ts, y_data=l2_length, name="l2", fix_scale=False)

animator = Animator(fig=fig, interval=30, total_time=T, animatable_datas=[segments, l1_log, l2_log], speed=1)

anim = FuncAnimation(animator.fig,
                     animator.animate,
                     frames=animator.num_frames,
                     interval=animator.interval)

plt.show()
