"""
Trail-tracking environment

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces
from stable_baselines3.common.noise import NormalActionNoise


class TrailEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    heading_bound = np.pi / 2
    max_speed = 2
    view_distance = 25
    max_steps = 20

    def __init__(self, trail_map):
        super().__init__()

        """
        The action space is the tuple (heading, velocity).
            Heading refers to the change in direction the agent is heading,
                in radians. North is equivalent to heading=0
            Velocity is the step-size the agent progresses in the environment
        """
        self.action_space = spaces.Box(
            low=np.array([-1]),
            high=np.array([1]))

        # self.action_space = spaces.Box(
        #     low=np.array([-1, -1]),
        #     high=np.array([1, 1]))

        """
        Observe the strength of odor in an ego-centric frame of reference. This
        space can be interpreted as a 2 * view_distance x 2 * view_distance x num_channels
        images.
            The first channel represents the agent's location history
            The second channel represents the agent's odor history
        """
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(2 * TrailEnv.view_distance, 2 * TrailEnv.view_distance, 3),
                                            dtype=np.uint8)

        self.map = trail_map
        self.agent = TrailAgent(self.map, TrailEnv.view_distance)
        self.curr_step = 0

    def step(self, action):
        # return observation, reward, done, info
        # self.agent.move(TrailEnv.heading_bound * action[0], TrailEnv.max_speed * action[1])
        # self.agent.move(TrailEnv.heading_bound * action[0], TrailEnv.max_speed * action[1])
        self.agent.move(TrailEnv.heading_bound * action[0], TrailEnv.max_speed)
        # self.agent.move_direct(TrailEnv.max_speed * action[0], TrailEnv.max_speed * action[1])

        self.agent.sniff()

        obs = self.agent.make_observation()
        reward, is_done = self.agent.get_reward()

        print('action: ', action, 'rew:', reward, 'pos:', self.agent.position)

        if self.curr_step == TrailEnv.max_steps:
            is_done = True
            print('hit max')

        if self.agent.position[0] > self.agent.view_distance or self.agent.position[0] < -self.agent.view_distance \
                or self.agent.position[1] > self.agent.view_distance or self.agent.position[1] < -self.agent.view_distance:
            is_done = True
            reward = -10
            print('Walked off!')

        self.curr_step += 1

        return obs, reward, is_done, {}

    def reset(self):
        # return observation  # reward, done, info can't be included
        self.curr_step = 0
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

    def move_direct(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy
        self.position_history.append(self.position[:])

    def sniff(self):
        odor = self.map.get_odor(*self.position)
        self.odor_history.append((odor, *self.position[:]))
        return odor

    # TODO: calibrate reward carefully <-- YES!
    def get_reward(self) -> Tuple[float, bool]:
        reward, is_done = self.map.get_reward(*self.position)
        # if not is_done:
        #     if self.position[1] - self.position_history[-2][1] > 0:
        #         reward = [1]
        #     else:
        #         reward = [-1]

        reward = 3 * (self.odor_history[-1][0] - self.odor_history[-2][0])

        if is_done:
            reward = [10]

        return reward[0], is_done

    def make_observation(self):
        pos_obs = self.make_pos_observation()
        odor_obs = self.make_odor_observation()

        self_obs = np.zeros((2 * self.view_distance, 2 * self.view_distance))
        x = int(np.round(self.position[0] + self.view_distance))
        y = int(np.round(self.position[1] + self.view_distance))

        if 0 <= x < self.view_distance * 2 \
                and 0 <= y < self.view_distance * 2:
            self_obs[x, y] = 255

        total_obs = np.stack((pos_obs, odor_obs, self_obs), axis=-1)
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

        # ego = (rot_trans @ (past_pos + orig_trans).T).T
        # ego = past_pos + orig_trans
        ego = past_pos
        ego_pos = ego + self.view_distance

        # Manhattan interpolation
        for i, point in enumerate(ego_pos[:-1]):
            next_point = ego_pos[i + 1]
            d = next_point - point
            steps = 2 * np.sum(np.abs(next_point - point)).astype('int')
            dx = (d[0] / steps) if steps != 0 else 0
            dy = (d[1] / steps) if steps != 0 else 0

            for i in range(steps):
                x_coord = np.round(point[0] + i * dx).astype(int)
                y_coord = np.round(point[1] + i * dy).astype(int)
                if 0 <= x_coord < self.view_distance * 2 \
                        and 0 <= y_coord < self.view_distance * 2:
                    pos_img[x_coord, y_coord] = 255

        return pos_img.astype(np.uint8)

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

        # ego = (rot_trans @ (past_pos + orig_trans).T).T
        # ego = past_pos + orig_trans
        ego = past_pos
        ego_pos = ego + self.view_distance

        for odor, pos in zip(past_odor, ego_pos):
            x_coord, y_coord = pos
            if 0 <= x_coord < self.view_distance * 2 - 1 \
                    and 0 <= y_coord < self.view_distance * 2 - 1:

                x = np.round(x_coord).astype(int)
                y = np.round(y_coord).astype(int)
                odor_img[x, y] = odor * 255

        return odor_img.astype(np.uint8)

    def render(self):
        self.map.plot()
        plt.plot(*self.position, 'ro')
        plt.show()


class TrailMap:
    def __init__(self):
        self.upper_left = (-50, 50)
        self.lower_right = (50, -50)
        self.resolution = 100

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
        print(np.max(odors))
        plt.scatter(xx.ravel(), yy.ravel(), odors.ravel())
        plt.scatter(*self.target)


class StraightTrail(TrailMap):
    def __init__(self):
        super().__init__()
        self.dist = 7
        self.target = (0, self.dist)
        self.tolerance = 3

    def get_odor(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array([x])
            y = np.array([y])

        odor = - 0.1 * ((x - self.target[0]) ** 2 + (y - self.target[1]) ** 2) + 10

        # odor = (y + 10) / (1 + 0.5 * x**2)
        # # odor = (y + 10)
        # odor[odor < 0] = 0
        # odor[y > self.dist] = 0

        return odor / 10

    def get_reward(self, x, y):

        is_done = np.all(np.isclose(self.target, (x, y), atol=self.tolerance))
        # is_done = np.isclose(self.target[1], y, atol=2) and np.isclose(self.target[0], x, atol=np.inf)
        # if is_done:
        #     reward = 10
        # else:
        #     reward = 0
        return self.get_odor(x, y), bool(is_done)

        if is_done:
            print("Success!")

        return [reward], bool(is_done)


# <codecell>
if __name__ == '__main__':

    env = TrailEnv(StraightTrail())
    # TODO: try CNN policy with real images
    model = DDPG("CnnPolicy", env,
                 verbose=1,
                 action_noise=NormalActionNoise(0, 1),
                 learning_starts=500)
    model.learn(total_timesteps=1000, log_interval=5)
    model.save('trail_model')
    exit()

# <codecell>
env = TrailEnv(StraightTrail())  # TODO: seems to be stuck in predicting the same thing
model = DDPG.load('trail_model')

obs = env.reset()
plt.plot(obs[..., 0])
for _ in range(10):
    plt.imshow(obs)
    action, _ = model.predict(obs)
    print(action)
    obs, _, is_done, _ = env.step(action)
    print(is_done)
    plt.show()

# <codecell>
env = TrailEnv(StraightTrail())

obs = env.reset()
plt.imshow(obs[..., 0])
plt.show()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, _, _ = env.step(action)
    print(reward)
    plt.imshow(obs)
    plt.show()


# <codecell>
env = TrailEnv(StraightTrail())
check_env(env)
print('done!')

# <codecell>

trail_map = StraightTrail()
agent = TrailAgent(trail_map, view_distance=25)
# print(agent.get_reward())
agent.move(0, 5)
agent.sniff()
print(agent.get_reward())
agent.move(np.pi / 4, 5)
odor = agent.sniff()
print(agent.get_reward())
agent.move(-np.pi / 2, 5)
agent.sniff()
print(agent.get_reward())
print(agent.odor_history)
# agent.move(-np.pi / 4, 0)
# agent.move(-np.pi / 6, 2)

obs = agent.make_observation()

img_corr = np.flip(obs.T, axis=0)
# print(img_corr)
plt.imshow(obs)

# %%
trail_map = StraightTrail()
trail_map.plot()
plt.show()

# %%
