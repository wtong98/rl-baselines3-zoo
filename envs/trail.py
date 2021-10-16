"""
Trail-tracking environment

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from ast import get_docstring
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces


class TrailEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    heading_bound = np.pi / 2
    max_speed = 10
    view_distance = 20

    def __init__(self, trail_map):
        super().__init__()

        """
        The action space is the tuple (heading, velocity).
            Heading refers to the change in direction the agent is heading,
                in radians. North is equivalent to heading=0
            Velocity is the step-size the agent progresses in the environment
        """

        self.action_space = spaces.Box(
            low=np.array([-TrailEnv.heading_bound, 0]),
            high=np.array([TrailEnv.heading_bound, TrailEnv.max_speed]))

        """
        Observe the strength of odor in an ego-centric frame of reference. This
        space can be interpreted as a 2 * view_distance x 2 * view_distance x num_channels
        images.
            The first channel represents the agent's location history
            The second channel represents the agent's odor history
        """
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2 * TrailEnv.view_distance, 2 * TrailEnv.view_distance, 2))

        self.map = trail_map
        self.agent = TrailAgent(self.map, TrailEnv.view_distance)

    def step(self, action):
        # return observation, reward, done, info
        self.agent.move(action[0], action[1])
        self.agent.sniff()

        obs = self.agent.make_observation()
        reward, is_done = self.agent.get_reward()

        return obs, reward, is_done, {}

    def reset(self):
        # return observation  # reward, done, info can't be included
        self.agent = TrailAgent(self.map, TrailEnv.view_distance)
        obs = self.agent.make_observation()
        return obs

    def render(self, mode='human'):
        obs = self.agent.make_observation()
        print(obs[:, :, 0])

    # def close(self):
    #     ...


class TrailAgent:
    def __init__(self, trail_map, view_distance):
        self.position = [0, 0]
        self.heading = 0
        self.map = trail_map
        self.view_distance = view_distance

        self.position_history = [[0, 0]]
        self.odor_history = []

        self.sniff()

    def move(self, d_heading, speed):
        self.heading += d_heading
        dx = np.sin(self.heading) * speed
        dy = np.cos(self.heading) * speed

        self.position[0] += dx
        self.position[1] += dy
        self.position_history.append(self.position[:])

    def sniff(self):
        odor = self.map.get_odor(*self.position)
        self.odor_history.append((odor, *self.position[:]))
        return odor

    def get_reward(self) -> Tuple[float, bool]:
        reward, is_done = self.map.get_reward(*self.position)
        return reward[0], is_done

    def make_observation(self):
        pos_obs = self.make_pos_observation()
        odor_obs = self.make_odor_observation()
        total_obs = np.stack((pos_obs, odor_obs), axis=-1)
        return total_obs

    def make_pos_observation(self):
        pos_img = np.zeros((2 * self.view_distance, 2 * self.view_distance))
        past_pos = np.vstack(self.position_history)

        orig_trans = -np.tile(self.position, (len(self.position_history), 1))
        rot_ang = self.heading
        rot_trans = np.array([
            [np.cos(rot_ang), -np.sin(rot_ang)],
            [np.sin(rot_ang), np.cos(rot_ang)]
        ])

        ego = (rot_trans @ (past_pos + orig_trans).T).T
        ego_pos = ego + self.view_distance

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
                if 0 <= x_coord < self.view_distance * 2 \
                        and 0 <= y_coord < self.view_distance * 2:
                    pos_img[x_coord, y_coord] = 1

        return pos_img

    def make_odor_observation(self):
        odor_img = np.zeros((2 * self.view_distance, 2 * self.view_distance))
        past = np.vstack(self.odor_history)
        past_odor = past[:, 0]
        past_pos = past[:, 1:]

        orig_trans = -np.tile(self.position, (len(past_pos), 1))
        rot_ang = self.heading
        rot_trans = np.array([
            [np.cos(rot_ang), -np.sin(rot_ang)],
            [np.sin(rot_ang), np.cos(rot_ang)]
        ])

        ego = (rot_trans @ (past_pos + orig_trans).T).T
        ego_pos = ego + self.view_distance

        for odor, pos in zip(past_odor, ego_pos):
            x_coord, y_coord = pos
            if 0 <= x_coord < self.view_distance * 2 \
                    and 0 <= y_coord < self.view_distance * 2:

                x = np.round(x_coord).astype(int)
                y = np.round(y_coord).astype(int)
                odor_img[x, y] = odor

        return odor_img

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

    def get_reward(self, x, y) -> Tuple[float, bool]:
        raise NotImplementedError('get_reward not implemented!')

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
        self.target = (0, 50)
        self.tolerance = 0.5

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

    def get_reward(self, x, y):
        is_done = np.all(np.isclose(self.target, (x, y), atol=0.5))
        return self.get_odor(x, y), bool(is_done)


if __name__ == '__main__':

    # from stable_baselines3.common.env_checker import check_env
    # env = TrailEnv(StraightTrail())
    # check_env(env)
    # print('done!')

    # trail_map = StraightTrail()
    # agent = TrailAgent(trail_map)
    # agent.move(0, 5)
    # agent.move(np.pi / 4, 5)
    # agent.move(np.pi / 4, 5)
    # agent.move(np.pi / 4, -5)
    # # agent.move(-np.pi / 4, 0)
    # # agent.move(-np.pi / 6, 2)

    # obs = agent.make_pos_observation(10)

    # img_corr = np.flip(obs.T, axis=0)
    # # print(img_corr)
    # plt.imshow(img_corr)

    # %%
