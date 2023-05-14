import numpy as np
import scipy.integrate as integrate
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
        self.current_time = 0

    def dstate_dt(self, state, t):
        """derivative of state at time t"""
        X = state[0:2]
        dX = state[2:4]
        return np.array([
            *dX,
            *self.g * normalize(self.gravity_center - X)
        ])

    def get_position(self):
        return self.state[0:2]

    def step(self, dt):
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.current_time += dt
