"""
Various trail maps and adventures:

author: William Tong
"""

# <codecell>
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class TrailMap:
    def __init__(self, start=None, end=None):
        self.start = start if start != None else np.array([0, 0])
        self.end = end if end != None else np.array([0, 0])
        self.tol = 2

    def sample(self, x, y):
        """ Returns odor on scale from 0 to 1 """
        raise NotImplementedError('sample not implemented!')

    def plot(self):
        raise NotImplementedError('plot not implemented!')

    def reset(self):
        raise NotImplementedError('reset not implemented!')

    def is_done(self, x, y):
        is_done = np.all(np.isclose(self.end, (x, y), atol=self.tol))
        return bool(is_done)


class StraightTrail(TrailMap):
    def __init__(self, end=None):
        super().__init__()
        self.end = end if end != None else np.array([10, 15])

    def sample(self, x, y):
        eps = 1e-8
        total_dist = np.sqrt((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2)
        perp_dist = np.abs((self.end[0] - self.start[0]) * (self.start[1] - y) - (self.start[0] - x) *
                           (self.end[1] - self.start[1])) / np.sqrt((self.start[0] - self.end[0]) ** 2 + (self.start[1] - self.end[1])**2 + eps)

        max_odor = np.sqrt(np.sum((self.end - self.start) ** 2)) + 1
        odor = max_odor - total_dist
        odor *= 1 / (perp_dist + 1)

        odor = np.clip(odor, 0, np.inf)
        return odor / max_odor

    def plot(self):
        x = np.linspace(-20, 20, 100)
        y = np.linspace(-20, 20, 100)
        xx, yy = np.meshgrid(x, y)

        odors = self.sample(xx, yy)
        plt.contourf(x, y, odors)
        plt.colorbar()

    def reset(self):
        pass


class RandomStraightTrail(StraightTrail):
    def __init__(self):
        super().__init__()
        self.end = self._rand_coords()
        self.tol = 4

    def _rand_coords(self):
        # new_end = np.random.randint(10, 16, 2) * np.random.choice([-1, 1], 2)
        # new_end = np.random.randint(15, 16, 2) * np.random.choice([-1, 1], 1)

        branch = np.random.choice([-1, 0, 1])
        x_coord = branch * 15
        new_end = np.array([x_coord, 15])

        # print(new_end)
        return new_end

    def reset(self):
        self.end = self._rand_coords()


class RoundTrail(TrailMap):
    def __init__(self):
        super().__init__()
        self.end = np.array([10, 15])

    def sample(self, x, y):
        # if not isinstance(x, np.ndarray):
        #     x = np.array([x])
        #     y = np.array([y])
        max_odor = 100
        odor = - 0.1 * ((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2) + max_odor
        odor = np.clip(odor, 0, np.inf)
        return odor / max_odor

    def plot(self):
        xx, yy = np.meshgrid(
            np.linspace(-20, 20, 100),
            np.linspace(-20, 20, 100))

        odors = self.sample(xx, yy)
        odors = np.flip(odors, axis=0)
        plt.imshow(odors)

    def reset(self):
        pass


class RandomRoundTrail(RoundTrail):
    def __init__(self):
        super().__init__()
        self.end = self._rand_coords()

    def _rand_coords(self):
        new_end = np.random.randint(10, 16, 2) * np.random.choice([-1, 1], 2)
        return new_end

    def reset(self):
        self.end = self._rand_coords()


class TrainingTrailSet(TrailMap):  # TODO: convert to give trails that are underperforming < -- STOPPED HERE
    def __init__(self, trails: List[TrailMap]):
        super().__init__()
        self.trails = trails
        self.curr_trail = self._get_rand_trail()

    def _get_rand_trail(self):
        rand_idx = np.random.randint(len(self.trails))
        return self.trails[rand_idx]

    def sample(self, x, y):
        return self.curr_trail.sample(x, y)

    def plot(self):
        for trail in self.trails:
            trail.plot()

    def reset(self):
        self.curr_trail = self._get_rand_trail()


# TODO: try training on straight trail of increasing lengths
if __name__ == '__main__':
    trail = StraightTrail()
    trail.plot()

# %%
