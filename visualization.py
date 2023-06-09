import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import interp1d

from matplotlib.animation import FuncAnimation
from abc import abstractmethod


# matplotlib config
mpl.rcParams['axes.formatter.useoffset'] = False


def interpolate(array, new_size):
    func = interp1d(np.linspace(0, 1, len(array)), array, axis=0)
    return func(np.linspace(0, 1, new_size))


class AnimatableData:
    @abstractmethod
    def set_frame(self, i):
        pass

    @abstractmethod
    def set_num_frames(self, num_frames):
        pass


class AnimatableKSegmentData(AnimatableData):
    # list of history of coordinates
    def __init__(self, ax: plt.Axes, points_data, name: str, fix_scale: bool = True):
        self.cur_line, = ax.plot([], [], 'r')

        self.points_data = np.array(points_data)

        mnx = self.points_data[:, :, 0].min()
        mxx = self.points_data[:, :, 0].max()
        mny = self.points_data[:, :, 1].min()
        mxy = self.points_data[:, :, 1].max()

        ax.legend([name])
        ax.set_xlim(mnx, mxx)
        ax.set_ylim(mny, mxy)
        if fix_scale:
            x_range = mxx - mnx
            y_range = mxy - mny
            ax.set_box_aspect(y_range / x_range)

    def set_num_frames(self, num_frames):
        self.points_data = interpolate(self.points_data, num_frames)

    def set_frame(self, i):
        self.cur_line.set_data(*self.points_data[i].swapaxes(0, 1))
        return self.cur_line,


class AnimatableSegmentData(AnimatableData):
    def __init__(self, ax: plt.Axes, p1_data, p2_data, name: str, fix_scale: bool = True):
        self.cur_line, = ax.plot([], [], 'r')

        self.p1_data = np.array(p1_data)
        self.p2_data = np.array(p2_data)

        mnx = min(*self.p1_data[:, 0], *self.p2_data[:, 0])
        mxx = max(*self.p1_data[:, 0], *self.p2_data[:, 0])
        mny = min(*self.p1_data[:, 1], *self.p2_data[:, 1])
        mxy = max(*self.p1_data[:, 1], *self.p2_data[:, 1])

        ax.legend([name])
        ax.set_xlim(mnx, mxx)
        ax.set_ylim(mny, mxy)
        if fix_scale:
            x_range = mxx - mnx
            y_range = mxy - mny
            ax.set_box_aspect(y_range / x_range)

    def set_num_frames(self, num_frames):
        self.p1_data = np.concatenate([
            interpolate(self.p1_data[:, 0], num_frames)[:, None],
            interpolate(self.p1_data[:, 1], num_frames)[:, None]], axis=1)
        self.p2_data = np.concatenate([
            interpolate(self.p2_data[:, 0], num_frames)[:, None],
            interpolate(self.p2_data[:, 1], num_frames)[:, None]], axis=1)

    def set_frame(self, i):
        self.cur_line.set_data([self.p1_data[i, 0], self.p2_data[i, 0]],
                               [self.p1_data[i, 1], self.p2_data[i, 1]])
        return self.cur_line,


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
    def __init__(self, fig, interval, total_time, animatable_datas: ["AnimatableData"], speed=1):
        self.fig = fig
        self.animatable_datas = animatable_datas
        self.interval = interval
        self.num_frames = round((total_time/speed) * 1000 / interval)
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

    animator = Animator(fig=fig, interval=30, total_time=1, animatable_datas=[obj1])
    anim = FuncAnimation(animator.fig,
                         animator.animate,
                         frames=animator.num_frames,
                         interval=animator.interval)
    plt.show()
