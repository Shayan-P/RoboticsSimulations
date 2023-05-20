import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from visualization import Animator, AnimatableLinePlot, AnimatableSegmentData


class Rod:
    def __init__(self,
                 center_pos=[10, 10],  # pos: x z
                 center_theta=0,
                 d_center_pos=[0, 3],
                 d_center_theta=0,
                 h_floor=0,
                 L=1.0,
                 mass=1.0,
                 inertia=1.0,
                 lamda=0.8,
                 g=9.8):
        self.state = np.array([*center_pos, center_theta, *d_center_pos, d_center_theta])
        self.L = L
        self.mass = mass
        self.inertia = inertia
        self.h_floor = h_floor
        self.g = g
        self.lamda = lamda

    def dstate_dt(self, t, state):
        """derivative of state at time t"""
        return np.array([
            *state[3:5], # d_center_pos
            state[5], # d_center_theta
            0, -self.g, # dd_center_pos
            0  # dd_center_theta
        ])

    def jacobian(self, state):
        """d [pos1, pos2] / d state"""
        theta = state[2]
        return np.array([
            [1, 0, +self.L / 2 * np.sin(theta)],  # pos1 x
            [0, 1, -self.L / 2 * np.cos(theta)],  # pos1 z
            [1, 0, -self.L / 2 * np.sin(theta)],  # pos2 x
            [0, 1, +self.L / 2 * np.cos(theta)]   # pos2 z
        ])

    def get_pos1(self):
        center = self.state[0:2]
        theta = self.state[2]
        return center - self.L / 2 * np.array([np.cos(theta), np.sin(theta)])

    def get_pos2(self):
        center = self.state[0:2]
        theta = self.state[2]
        return center + self.L / 2 * np.array([np.cos(theta), np.sin(theta)])

    def get_theta(self):
        return self.state[2]

    def get_d_theta(self):
        return self.state[5]

    def get_center_speed(self):
        return self.state[3:5]

    def event(self, t, state):
        """a continuous function that its change in sign signals ode to stop progressing"""
        centerz = state[1]
        theta = state[2]
        z1 = centerz - self.L / 2 * np.sin(theta)
        z2 = centerz + self.L / 2 * np.sin(theta)
        return min(z1, z2) - self.h_floor

    def apply_ground_impulse(self):
        # switch
        # governing equations:
        #       M ddq = J^T F_tip    ->  M dq_post - J^T Impulse = M dq
        #       dp_post = -lambda * dp -> J dq_post = -lambda J dq
        #      [[M   -J^T]    [dq_post    =  [ M dq
        #       [J      0]]    Impulse]   =    -lambda J dq ]
        ########
        M = np.array([[self.mass, 0, 0],
                      [0, self.mass, 0],
                      [0, 0, self.inertia]])
        J = self.jacobian(self.state)
        if self.get_pos1()[1] < self.get_pos2()[1]:  ## pos1 lower
            J = np.array([J[1]])
        else:                                                            ## pos2 lower
            J = np.array([J[3]])
        dq = self.state[3:].reshape(3, 1)
        A = np.block([
            [M, -J.T],
            [J, np.zeros((J.shape[0], J.shape[0]))]])
        b = np.block([
            [M @ dq],
            [-self.lamda * J @ dq]
        ])
        inversed = np.linalg.inv(A) @ b
        dq_post = inversed[0:3]
        self.state = np.array([*self.state[0:3], *dq_post.reshape((3,))])

    def step(self, dt):
        if dt <= 1e-9:
            return
        new_state = integrate.odeint(self.dstate_dt, self.state, [0, dt], tfirst=True)[1]
        if (self.event(dt, new_state) > 0) ^ (self.event(dt, self.state) > 0): # change in even
            # for now just ignore finding the exact time of the hit
            self.apply_ground_impulse()
        else:
            self.state = new_state


if __name__ == "__main__":
    fig, (ax_rod, ax_speed_log, ax_theta_log, ax_d_theta_log) = plt.subplots(4, 1)

    rod = Rod(
        center_pos=[0, 5],  # pos: x z
        center_theta=1,
        d_center_pos=[4, 0],
        d_center_theta=0,
        h_floor=0,
        L=3.0,
        mass=1.0,
        inertia=0.1,
        lamda=0.9,
        g=9.81)

    p1_data = []
    p2_data = []
    v_data = []
    theta_data = []
    dtheta_data = []
    t_data = []
    dt = 0.001
    total_time = 10
    cur_time = 0
    while cur_time < total_time:
        p1_data.append(rod.get_pos1())
        p2_data.append(rod.get_pos2())
        v_data.append(np.linalg.norm(rod.get_center_speed()))
        theta_data.append(rod.get_theta())
        dtheta_data.append(rod.get_d_theta())
        t_data.append(cur_time)

        cur_time += dt
        rod.step(dt)

    animPoint = AnimatableSegmentData(ax_rod, p1_data=p1_data, p2_data=p2_data, name="rod", fix_scale=True)
    animSpeedLog = AnimatableLinePlot(ax_speed_log, x_data=t_data, y_data=v_data, name="speed center", fix_scale=False)
    animThetaLog = AnimatableLinePlot(ax_theta_log, x_data=t_data, y_data=v_data, name="theta", fix_scale=False)
    animDThetaLog = AnimatableLinePlot(ax_d_theta_log, x_data=t_data, y_data=v_data, name="speed theta", fix_scale=False)

    animator = Animator(fig=fig, interval=30, total_time=total_time, animatable_datas=[animPoint, animSpeedLog,
                                                                                       animThetaLog, animDThetaLog],
                        speed=1.0)

    anim = FuncAnimation(animator.fig,
                         animator.animate,
                         frames=animator.num_frames,
                         interval=animator.interval)
    plt.show()
