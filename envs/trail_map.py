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
    def __init__(self, end=None, narrow_factor=1):
        super().__init__()
        self.end = end if type(end) != type(None) else np.array([10, 15])
        self.narrow_factor = narrow_factor

    def sample(self, x, y):
        eps = 1e-8
        total_dist = np.sqrt((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2)
        perp_dist = np.abs((self.end[0] - self.start[0]) * (self.start[1] - y) - (self.start[0] - x) *
                           (self.end[1] - self.start[1])) / np.sqrt((self.start[0] - self.end[0]) ** 2 + (self.start[1] - self.end[1])**2 + eps)

        max_odor = np.sqrt(np.sum((self.end - self.start) ** 2)) + 1
        odor = max_odor - total_dist
        odor *= 1 / (perp_dist + 1) ** self.narrow_factor

        # odor = 1 / (perp_dist + 1) ** self.narrow_factor
        # max_dist = np.sqrt(np.sum((self.end - self.start) ** 2))
        # if np.isscalar(total_dist):
        #     if total_dist > max_dist:
        #         odor *= 1 / (total_dist - max_dist + 1) ** self.narrow_factor
        # else:
        #     adjust = 1 / (np.clip(total_dist - max_dist, 0, np.inf) + 1) ** self.narrow_factor
        #     odor *= adjust

        odor = np.clip(odor, 0, np.inf)
        return odor / max_odor
        return odor

    def plot(self, ax=None):
        x = np.linspace(-20, 20, 100)
        y = np.linspace(-20, 20, 100)
        xx, yy = np.meshgrid(x, y)

        odors = self.sample(xx, yy)

        if ax:
            return ax.contourf(x, y, odors)
        else:
            plt.contourf(x, y, odors)
            plt.colorbar()

    def reset(self):
        pass


class RandomStraightTrail(StraightTrail):
    def __init__(self, is_eval=False, **kwargs):
        super().__init__(**kwargs)
        self.eval = is_eval
        self.next_choice = 0

        self.end = self._rand_coords()
        self.tol = 4

    def _rand_coords(self):
        # branches = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0)]
        branches = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)]

        if self.eval:
            idx = self.next_choice
            self.next_choice = (self.next_choice + 1) % len(branches)
        else:
            idx = np.random.choice(len(branches))

        x_fac, y_fac = branches[idx]
        new_end = np.array([x_fac * 15, y_fac * 15])

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


class TrainingTrailSet(TrailMap):  # TODO: test
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
    trail = StraightTrail(narrow_factor=2)
    trail.plot()

# %%
