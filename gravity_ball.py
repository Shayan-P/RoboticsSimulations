import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from visualization import AnimatablePointData, AnimatableLinePlot, Animator
from utils import normalize


class GravityBall:
    def __init__(self,
                 pos=[10, 10],
                 d_pos=[3, 4],
                 gravity_center=[0, 0],
                 mass=1.0,
                 g=1.0
                 ):
        self.state = np.array([*pos, *d_pos])
        self.g = g
        self.mass = mass
        self.gravity_center = np.array(gravity_center)

    def dstate_dt(self, t, state):
        """derivative of state at time t"""
        X = state[0:2]
        dX = state[2:4]
        return np.array([
            *dX,
            *self.g * normalize(self.gravity_center - X)
        ])

    def get_position(self):
        return self.state[0:2]

    def get_speed(self):
        return self.state[2:4]

    def step(self, dt):
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt], tfirst=True)[1]


if __name__ == "__main__":
    fig, (ax_ball, ax_log) = plt.subplots(2, 1)

    ball = GravityBall(
        pos=[10, 0],
        d_pos=[-1, 7],
        gravity_center=[0, 0],
        mass=1.0,
        g=10.0
    )

    x_data = []
    y_data = []
    v_data = []
    t_data = []
    dt = 0.001
    total_time = 10
    cur_time = 0
    while cur_time < total_time:
        x_data.append(ball.get_position()[0])
        y_data.append(ball.get_position()[1])
        v_data.append(np.linalg.norm(ball.get_speed()))
        t_data.append(cur_time)

        cur_time += dt
        ball.step(dt)

    animPoint = AnimatablePointData(ax_ball, x_data=x_data, y_data=y_data, name="gravity ball", fix_scale=True)
    animLog = AnimatableLinePlot(ax_log, x_data=t_data, y_data=v_data, name="speed", fix_scale=False)

    animator = Animator(fig=fig, interval=30, total_time=total_time, animatable_datas=[animPoint, animLog], speed=1)
    anim = FuncAnimation(animator.fig,
                         animator.animate,
                         frames=animator.num_frames,
                         interval=animator.interval)
    plt.show()
