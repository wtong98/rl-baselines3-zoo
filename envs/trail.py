"""
Trail-tracking environment

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces


class TrailEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    heading_bound = np.pi / 2
    max_speed = 10
    view_distance = 20

    def __init__(self):
        super().__init__()

        """
        The action space is the tuple (heading, velocity).
            Heading refers to the change in direction the agent is heading,
                in radians. North is equivalent to heading=0
            Velocity is the step-size the agent progresses in the environment
        """

        self.action_space = spaces.Box(
            low=np.array([-TrailEnv.heading_bound, 0]),
            high=np.array([TrailEnv.heading_bound, TrailEnv.max_speed]),
            shape=(2,))

        """
        Observe the strength of odor in an ego-centric frame of reference. This
        space can be interpreted as a 2 * view_distance x 2 * view_distance x num_channels
        images.
            The first channel represents the agent's location history
            The second channel represents the agent's odor history
        """
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(-TrailEnv.view_distance, TrailEnv.view_distance, 2))

    def step(self, action):
        # return observation, reward, done, info
        return 0, 0, False, None

    def reset(self):
        # return observation  # reward, done, info can't be included
        return 0

    # def render(self, mode='human'):
    #     ...

    # def close(self):
    #     ...


class TrailAgent:
    def __init__(self, trail_map):
        self.position = [0, 0]
        self.map = trail_map

        self.position_history = [[0, 0]]
        self.odor_history = []

    def move(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy
        self.position_history.append(self.position[:])

    def sniff(self):
        odor = self.map.get_odor(*self.position)
        self.odor_history.append((odor, *self.position[:]))

    def make_pos_observation(self, heading, view_distance):
        pos_img = np.zeros((2 * view_distance, 2 * view_distance))
        past_pos = np.vstack(self.position_history)

        orig_trans = -np.tile(self.position, (len(self.position_history), 1))
        rot_ang = heading
        rot_trans = np.array([
            [np.cos(rot_ang), -np.sin(rot_ang)],
            [np.sin(rot_ang), np.cos(rot_ang)]
        ])

        ego = (rot_trans @ (past_pos + orig_trans).T).T
        ego_pos = ego + view_distance

        # Manhattan interpolation
        for i, point in enumerate(ego_pos[:-1]):
            next_point = ego_pos[i + 1]
            d = next_point - point
            steps = 2 * np.sum(np.abs(next_point - point)).astype('int')
            dx = (d[0] / steps)
            dy = (d[1] / steps)

            for i in range(steps):
                x_coord = np.round(point[0] + i * dx).astype(int)
                y_coord = np.round(point[1] + i * dy).astype(int)
                if 0 <= x_coord < view_distance * 2 \
                        and 0 <= y_coord < view_distance * 2:
                    pos_img[x_coord, y_coord] = 1

        return pos_img

    def make_odor_observation(self, heading, view_distance):
        odor_img = np.zeros((2 * view_distance, 2 * view_distance))
        past = np.vstack(self.odor_history)
        past_odor = past[:, 0]
        past_pos = past[:, 1:]

        orig_trans = -np.tile(self.position, (len(self.position_history), 1))
        rot_ang = heading
        rot_trans = np.array([
            [np.cos(rot_ang), -np.sin(rot_ang)],
            [np.sin(rot_ang), np.cos(rot_ang)]
        ])

        ego = (rot_trans @ (past_pos + orig_trans).T).T
        ego_pos = ego + view_distance

        for odor, pos in zip(past_odor, ego_pos):
            x_coord, y_coord = pos
            if 0 <= x_coord < view_distance * 2 \
                    and 0 <= y_coord < view_distance * 2:
                odor_img[x_coord, y_coord] = odor

        return odor_img  # TODO: test

    def render(self):
        self.map.plot()
        plt.plot(*self.position, 'ro')
        plt.show()


class TrailMap:
    def __init__(self):
        self.upper_left = (-100, 100)
        self.lower_right = (100, -100)
        self.resolution = 200

    def get_odor(self, x, y):
        raise NotImplementedError('get_odor not implemented!')

    def size(self):
        return (self.resolution, self.resolution)

    def plot(self):
        xx, yy = np.meshgrid(
            np.linspace(self.upper_left[0], self.lower_right[0], self.resolution),
            np.linspace(self.upper_left[1], self.lower_right[1], self.resolution))

        odors = self.get_odor(xx, yy)
        plt.scatter(xx.ravel(), yy.ravel(), odors.ravel())


class StraightTrail(TrailMap):
    def __init__(self):
        super().__init__()

    def get_odor(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array([x])
            y = np.array([y])

        vert_dist = 50 - y
        vert_dist[y < 0] = 0
        vert_dist[y > 50] = 0

        odor = np.ones(x.shape)
        odor[y < 0] = 0
        odor[y > 50] = 0

        odor -= vert_dist / 50

        horiz_dist = np.abs(x)
        horiz_scale = 1 / (1 + 2 * horiz_dist)
        horiz_scale[horiz_scale < 0.2] = 0
        odor *= horiz_scale

        return odor


if __name__ == '__main__':
    trail_map = StraightTrail()
    agent = TrailAgent(trail_map)
    agent.move(0, 5)
    agent.move(5, 5)
    agent.move(-5, 5)
    agent.move(-5, -5)
    agent.move(5, -5)
    obs = agent.make_observation(0, 15)

    img_corr = np.flip(obs.T, axis=0)
    # print(img_corr)
    plt.imshow(img_corr)

# %%
