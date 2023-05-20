import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from abc import abstractmethod


def interpolate(array, new_size):
    return np.interp(np.linspace(0, 1, new_size),
                     np.linspace(0, 1, len(array)),
                     array)


class AnimatableData:
    @abstractmethod
    def set_frame(self, i):
        pass

    @abstractmethod
    def set_num_frames(self, num_frames):
        pass


class AnimatablePointData(AnimatableData):
    def __init__(self, ax: plt.Axes, x_data, y_data, name: str, fix_scale: bool = True):
        self.hist_line, = ax.plot([], [], 'g-')
        self.cur_line, = ax.plot([], [], 'ro')

        self.x_data = x_data
        self.y_data = y_data

        ax.legend([name + "_hist", name + "_curr"])
        ax.set_xlim(min(x_data), max(x_data))
        ax.set_ylim(min(y_data), max(y_data))
        if fix_scale:
            x_range = max(x_data) - min(x_data)
            y_range = max(y_data) - min(y_data)
            ax.set_box_aspect(y_range / x_range)

    def set_num_frames(self, num_frames):
        self.x_data = interpolate(self.x_data, num_frames)
        self.y_data = interpolate(self.y_data, num_frames)

    def set_frame(self, i):
        self.hist_line.set_data(self.x_data[:i+1], self.y_data[:i+1])
        self.cur_line.set_data([self.x_data[i]], [self.y_data[i]])
        return self.cur_line, self.hist_line


class AnimatableLinePlot(AnimatableData):
    def __init__(self, ax: plt.Axes, x_data, y_data, name: str, fix_scale: bool, plot_config="r"):
        self.line, = ax.plot([], [], plot_config)
        self.x_data = x_data
        self.y_data = y_data

        ax.legend([name])
        ax.set_xlim(min(x_data), max(x_data))
        ax.set_ylim(min(y_data), max(y_data))
        if fix_scale:
            x_range = max(x_data) - min(x_data)
            y_range = max(y_data) - min(y_data)
            ax.set_box_aspect(y_range / x_range)

    def set_num_frames(self, num_frames):
        self.x_data = interpolate(self.x_data, num_frames)
        self.y_data = interpolate(self.y_data, num_frames)

    def set_frame(self, i):
        self.line.set_data(self.x_data[:i+1], self.y_data[:i+1])
        return self.line,


class Animator:
    def __init__(self, fig, num_frames, total_time, animatable_datas: ["AnimatableData"]):
        self.fig = fig
        self.animatable_datas = animatable_datas
        self.num_frames = num_frames
        self.interval = total_time * 1000 / num_frames  # in ms
        for animatable in self.animatable_datas:
            animatable.set_num_frames(self.num_frames)

    def animate(self, i):
        animators = []
        for animatable in self.animatable_datas:
            animators.extend(animatable.set_frame(i))
        return tuple(animators)


if __name__ == "__main__":
    # sample use-case
    fig, ax = plt.subplots()

    obj1 = AnimatableLinePlot(ax, x_data=[1, 2, 3, 4, 5], y_data=[5, 1, 2, 4, 10], name="random", fix_scale=True,
                              plot_config="r-")
    obj2 = AnimatablePointData(ax, x_data=[4, 10, 1, 3, 2], y_data=[5, 1, 2, 4, 10], name="ball", fix_scale=True)

    animator = Animator(fig=fig, num_frames=40, total_time=1, animatable_datas=[obj1])
    anim = FuncAnimation(animator.fig,
                         animator.animate,
                         frames=animator.num_frames,
                         interval=animator.interval)
    plt.show()
