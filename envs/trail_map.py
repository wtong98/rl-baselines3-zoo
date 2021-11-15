"""
Various trail maps and adventures:

author: William Tong
"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt


class TrailMap:
    def __init__(self, start=None, end=None):
        self.start = start if start != None else np.array([0, 0])
        self.end = end if end != None else np.array([0, 0])

    def sample(self, x, y):
        """ Returns odor on scale from 0 to 1 """
        raise NotImplementedError('sample not implemented!')

    def plot(self):
        raise NotImplementedError('plot not implemented!')

    def is_done(self, x, y, tol=2):
        is_done = np.all(np.isclose(self.end, (x, y), atol=tol))
        return bool(is_done)


class StraightTrail(TrailMap):
    def __init__(self, end=None):
        super().__init__()
        self.end = end if end != None else np.array([10, 10])

    def sample(self, x, y):
        total_dist = np.sqrt((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2)
        perp_dist = np.abs((self.end[0] - self.start[0]) * (self.start[1] - y) - (self.start[0] - x) *
                           (self.end[1] - self.start[1])) / np.sqrt((self.start[0] - self.end[0]) ** 2 + (self.start[1] - self.end[1]))

        max_odor = np.sqrt(np.sum((self.end - self.start) ** 2)) + 1
        odor = max_odor - total_dist
        odor *= 1 / (perp_dist + 1)

        odor = np.clip(odor, 0, np.inf)
        return odor / max_odor

    def plot(self):
        xx, yy = np.meshgrid(
            np.linspace(-20, 20, 100),
            np.linspace(-20, 20, 100))

        odors = self.sample(xx, yy)
        odors = np.rot90(odors)
        plt.imshow(odors)


class RoundTrail(TrailMap):
    def __init__(self):
        super().__init__()
        self.end = np.array([10, 15])

    def sample(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array([x])
            y = np.array([y])

        odor = - 0.1 * ((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2) + 100
        return odor / 10

    def plot(self):
        xx, yy = np.meshgrid(
            np.linspace(-20, 20, 100),
            np.linspace(-20, 20, 100))

        odors = self.sample(xx, yy)
        odors = np.rot90(odors)
        plt.imshow(odors)


# TODO: try training on straight trail of increasing lengths
if __name__ == '__main__':
    trail = StraightTrail()
    trail.plot()

# %%
