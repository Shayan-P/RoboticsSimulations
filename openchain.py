import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from typing import TypeAlias
from tqdm import tqdm
from sympy import lambdify
from scipy import integrate
from visualization import AnimatableKSegmentData, Animator, AnimatableLinePlot


# simulation constants
class Params:
    def __init__(self, n: int, # number of links
                 ls: list[float], # length of each link
                 ms: list[float], # mass of each link
                 inertias: list[float], # inertia of each link
                 rcom: list[float], # ratio of center of mass. com is p[i] + (p[i+1] - p[i]) * rcom[i]
                 g: float # gravity
                 ):
        self.n = n
        self.ls = ls
        self.ms = ms
        self.inertias = inertias
        self.rcom = rcom
        self.g = g


Vector: TypeAlias = np.array


class Dynamics:
    def __init__(self, params: Params):
        self.params = params

    def forward_dynamics(self, q: Vector, dq: Vector, tau: Vector):
        """ calculates ddq/dt2 """
        raise NotImplementedError

    def forward_dynamics(self, q: Vector, dq: Vector, ddq: Vector):
        """ calculates tau to create ddq """
        raise NotImplementedError


class LagrangianDynamics(Dynamics):
    def __init__(self, params: Params):
        super().__init__(params)
        self.M, self.h = self.formulate_dynamics()
        # M(q) ddq + h(q, dq) = tau

    def formulate_dynamics(self):
        n = self.params.n

        q = np.array(smp.MatrixSymbol('q', n, 1)).reshape((n,))
        dq = np.array(smp.MatrixSymbol('dq', n, 1)).reshape((n,))
        rcom = self.params.rcom
        ls = self.params.ls
        ms = self.params.ms
        g = self.params.g
        inertias = self.params.inertias

        p_joint = np.zeros(2)
        K = smp.numer(0)
        U = smp.numer(0)

        for i in tqdm(range(n), 'construct kinematics'):
            qsum = q[0:i + 1].sum()
            vec = np.block([smp.cos(qsum) * ls[i],
                            smp.sin(qsum) * ls[i]]
                           )
            p_com = p_joint + rcom[i] * vec
            v_com_0 = sum(smp.diff(p_com[0], q[j]) * dq[j] for j in range(i + 1))  # jacobian
            v_com_1 = sum(smp.diff(p_com[1], q[j]) * dq[j] for j in range(i + 1))  # jacobian
            v_com = np.array([v_com_0, v_com_1])

            w_com = dq[0:i + 1].sum()

            U += ms[i] * g * p_com[1]
            K += 0.5 * ((v_com @ v_com) * ms[i] + w_com * w_com * inertias[i])

            p_joint = p_joint + vec

        K = K.simplify()

        M = [
            [K.diff(dq[i]).diff(dq[j]).simplify()
             for j in range(n)]
            for i in tqdm(range(n), 'calculate mass matrix')]  # M = ddK / dq^2

        C = [
            sum(
                (0.5 * (M[i][k].diff(q[j]) + M[i][j].diff(q[k]) - M[j][k].diff(q[i])) * dq[j] * dq[k]).simplify()
                for j in range(n) for k in range(n)
            )
            for i in tqdm(range(n), 'calculate coriolis matrix'
                                    'ix')]  # C = dqT Gamma dq
        G = [
            smp.diff(U, q[i])
            for i in tqdm(range(n), 'calculating g(theta)')
        ]

        M_lambified = [
            [lambdify([q], M[i][j], 'numpy')
             for j in range(n)]
            for i in range(n)
        ]
        M_func = lambda q: np.array([
            [M_lambified[i][j](q)
             for j in range(n)]
            for i in range(n)
        ])

        C_lambified = [lambdify([q, dq], C[i], 'numpy')
             for i in range(n)]
        G_lambified = [lambdify([q], G[i], 'numpy')
             for i in range(n)]
        h_func = lambda q, dq: np.array([
            G_lambified[i](q) + C_lambified[i](q, dq)
            for i in range(n)
        ])
        return M_func, h_func

    def forward_dynamics(self, q: Vector, dq: Vector, tau: Vector):
        """ calculates ddq/dt2 """
        # M(q) ddq + h(q, dq) = tau
        return np.linalg.solve(
            self.M(q),
            tau - self.h(q, dq)
        )

    def mass_matrix(self, q):
        return self.M(q)


class Kinematics:
    def __init__(self, params: Params):
        self.params = params

    def get_positions(self, q):  # join state
        n = self.params.n
        ls = self.params.ls
        p_joint = np.zeros(2)
        ps = [p_joint]
        for i in range(n):
            qsum = q[0:i + 1].sum()
            vec = np.block([np.cos(qsum) * ls[i],
                            np.sin(qsum) * ls[i]])
            p_joint = p_joint + vec
            ps.append(p_joint)
        return ps

    def get_com_positions(self, q):
        n = self.params.n
        ls = self.params.ls
        rs = self.params.rcom
        p_joint = np.zeros(2)
        p_coms = []
        for i in range(n):
            qsum = q[0:i + 1].sum()
            vec = np.block([np.cos(qsum) * ls[i],
                            np.sin(qsum) * ls[i]])
            p_com = p_joint + vec * rs[i]
            p_joint = p_joint + vec
            p_coms.append(p_com)
        return p_coms


def simulate_no_torque(params: Params, dt: float, duration: float, q0: Vector, dq0: Vector):
    cur_t = 0
    point_history = []
    mech_e_hist = []
    potential_e_hist = []
    kinetic_e_hist = []
    ts = []

    kinematics = Kinematics(params)
    dynamics = LagrangianDynamics(params)

    q = q0
    dq = dq0

    for _frame in tqdm(range(round(duration/dt)), "simulating open chain"):
        points = kinematics.get_positions(q)
        point_history.append(points)
        ts.append(cur_t)

        potential = sum(
            m * pos[1] * params.g
            for m, pos in zip(params.ms, kinematics.get_com_positions(q))
        )
        kinetic = 0.5 * (dq.T @ dynamics.mass_matrix(q) @ dq)
        potential_e_hist.append(potential)
        kinetic_e_hist.append(kinetic)
        mech_e_hist.append(potential + kinetic)

        zero_tau = np.zeros_like(q)
        # X = q, dq
        x_shape = np.array([q, dq]).shape  # we have to flatten it since ode only works with flat vectors

        def dstate_dt(t, X):
            q, dq = list(X.reshape(x_shape))
            dX = np.array([dq, dynamics.forward_dynamics(q, dq, zero_tau)])
            return dX.reshape((-1,))
        cur_t += dt
        x0 = np.array([q, dq]).reshape((-1,))
        x_new = integrate.odeint(dstate_dt, x0, [0, dt], tfirst=True)[1]
        q, dq = list(x_new.reshape(x_shape))

    return point_history, {
        "potential_e_hist": potential_e_hist,
        "mech_e_hist": mech_e_hist,
        "kinetic_e_hist": kinetic_e_hist,
        "ts": ts}


def plot_animation(points_history, logs, total_time, speed):
    # plot
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    spec = plt.GridSpec(ncols=1, nrows=4,
                             width_ratios=[1], wspace=0.5,
                             hspace=0.5, height_ratios=[4, 1, 1, 1])
    main_ax = fig.add_subplot(spec[0])
    u_ax = fig.add_subplot(spec[1])
    k_ax = fig.add_subplot(spec[2])
    e_ax = fig.add_subplot(spec[3])

    segments = AnimatableKSegmentData(main_ax, points_data=points_history, name="open chain", fix_scale=True)
    potential_e_plot = AnimatableLinePlot(u_ax, x_data=logs["ts"], y_data=logs["potential_e_hist"], name="potential energy", fix_scale=False)
    kinetic_e_plot = AnimatableLinePlot(k_ax, x_data=logs["ts"], y_data=logs["kinetic_e_hist"], name="kinetic energy", fix_scale=False)
    mech_e_plot = AnimatableLinePlot(e_ax, x_data=logs["ts"], y_data=logs["mech_e_hist"], name="mechanical energy", fix_scale=False)
    animator = Animator(fig=fig, interval=30, total_time=total_time,
                        animatable_datas=[segments, mech_e_plot, potential_e_plot, kinetic_e_plot],
                        speed=speed)

    anim = FuncAnimation(animator.fig,
                         animator.animate,
                         frames=animator.num_frames,
                         interval=animator.interval)
    plt.show()


if __name__ == "__main__":
    params = Params(
        n=3,
        ls=[1, 1, 1],
        ms=[1, 1, 1],
        inertias=[0.1, 0.1, 0.1],
        rcom=[0.3, 0.3, 0.3],
        g=9.8)

    q0 = np.array([0, np.pi/2, -np.pi/2])
    dq0 = np.array([0, 0, 0])

    total_time = 10
    points_history, logs = simulate_no_torque(params=params, dt=0.001, duration=total_time, q0=q0, dq0=dq0)
    points_history = np.array(points_history)
    plot_animation(points_history, logs, total_time=total_time, speed=1)
