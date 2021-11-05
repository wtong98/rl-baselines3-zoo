"""
Trail-tracking environment

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG, TD3, PPO
from typing import Tuple
import imageio
import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ActorCriticCnnPolicy


class TrailEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    heading_bound = np.pi
    max_speed = 3
    view_distance = 25
    max_steps = 20

    def __init__(self, trail_map=None, discrete=True):
        super().__init__()

        self.discrete = discrete

        if trail_map == None:
            trail_map = StraightTrail()

        """
        The action space is the tuple (heading, velocity).
            Heading refers to the change in direction the agent is heading,
                in radians. North is equivalent to heading=0
            Velocity is the step-size the agent progresses in the environment
        """

        if self.discrete:
            self.action_space = spaces.Discrete(8)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

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
        # old ideas:
        # self.agent.move(TrailEnv.heading_bound * action[0], TrailEnv.max_speed * action[1])
        # self.agent.move(0, action[0] * TrailEnv.max_speed)
        # self.agent.move_direct(TrailEnv.max_speed * action[0], TrailEnv.max_speed * action[1])

        if self.discrete:
            heading = (action / self.action_space.n) * 2 * np.pi
            self.agent.move_abs(heading, TrailEnv.max_speed)
        else:
            heading = action[0] * np.pi
            speed = ((action[1] + 1) / 2) * TrailEnv.max_speed
            self.agent.move_abs(heading, speed)

        self.agent.sniff()

        obs = self.agent.make_observation()
        reward, is_done = self.agent.get_reward()

        if self.curr_step == TrailEnv.max_steps:
            is_done = True
            # print('hit max')

        if self.agent.position[0] > self.agent.view_distance or self.agent.position[0] < -self.agent.view_distance \
                or self.agent.position[1] > self.agent.view_distance or self.agent.position[1] < -self.agent.view_distance:
            is_done = True
            reward = -10
            # print('Walked off!')

        self.curr_step += 1

        # print('action: ', action, 'rew:', reward, 'pos:', self.agent.position)
        return obs, reward, is_done, {}

    def reset(self):
        self.curr_step = 0
        self.agent = TrailAgent(self.map, TrailEnv.view_distance)
        obs = self.agent.make_observation()
        return obs

    def render(self, mode='human'):
        obs = self.agent.make_observation()
        print(obs[:, :, 0])

    def play_anim(self, model, is_deterministic=False, out_path='out.mp4'):
        obs = self.reset()
        frames = [obs]

        for _ in range(self.max_steps):
            action, _ = model.predict(obs, deterministic=is_deterministic)
            obs, _, is_done, _ = env.step(action)
            frames.append(obs)

            if is_done:
                break

        imageio.mimwrite(out_path, frames, fps=2)


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

    def move_abs(self, heading, speed):
        dx = np.sin(heading) * speed
        dy = np.cos(heading) * speed

        self.position[0] += dx
        self.position[1] += dy
        self.position_history.append(self.position[:])

    def sniff(self):
        odor = self.map.get_odor(*self.position)
        self.odor_history.append((odor, *self.position[:]))
        return odor

    def get_reward(self) -> Tuple[float, bool]:
        reward, is_done = self.map.get_reward(*self.position)

        # TODO: experiment with different reward strategies
        reward = 10 * (self.odor_history[-1][0] - self.odor_history[-2][0])

        if is_done:
            reward = [100]

        return reward[0], is_done

    def make_observation(self):
        pos_obs = self.make_pos_observation()
        odor_obs = self.make_odor_observation()

        self_obs = np.zeros((2 * self.view_distance, 2 * self.view_distance))
        # x = int(np.round(self.position[0] + self.view_distance))
        # y = int(np.round(self.position[1] + self.view_distance))

        # if 0 <= x < self.view_distance * 2 \
        #         and 0 <= y < self.view_distance * 2:
        #     self_obs[x, y] = 255

        # self_obs = np.flip(self_obs.T, axis=0)

        total_obs = np.stack((pos_obs, odor_obs, self_obs), axis=-1)
        # total_obs = np.stack((pos_obs, odor_obs), axis=-1)
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
        ego = past_pos + orig_trans
        # ego = past_pos
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

        pos_img = np.flip(pos_img.T, axis=0)
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
        ego = past_pos + orig_trans
        # ego = past_pos
        ego_pos = ego + self.view_distance

        for odor, pos in zip(past_odor, ego_pos):
            x_coord, y_coord = pos
            if 0 <= x_coord < self.view_distance * 2 - 1 \
                    and 0 <= y_coord < self.view_distance * 2 - 1:

                x = np.round(x_coord).astype(int)
                y = np.round(y_coord).astype(int)
                odor_img[x, y] = odor * 255

        odor_img = np.flip(odor_img.T, axis=0)
        return odor_img.astype(np.uint8)

    def render(self):
        self.map.plot()
        plt.plot(*self.position, 'ro')
        plt.show()


class TrailContinuousEnv(TrailEnv):

    def __init__(self, trail_map=None):
        super().__init__(trail_map=trail_map, discrete=False)


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


class RoundTrail(TrailMap):
    def __init__(self):
        super().__init__()
        self.target = (10, 15)
        self.tolerance = 2

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

    # TODO: consolidate rewards
    def get_reward(self, x, y):
        is_done = np.all(np.isclose(self.target, (x, y), atol=self.tolerance))
        # return self.get_odor(x, y), bool(is_done)
        return None, bool(is_done)


class StraightTrail(TrailMap):
    def __init__(self):
        super().__init__()
        self.target = (10, 15)
        self.tolerance = 2

    def get_odor(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array([x])
            y = np.array([y])

        odor = - 0.1 * ((x - self.target[0]) ** 2 + (y - self.target[1]) ** 2) + 100

        # odor = (y + 10) / (1 + 0.5 * x**2)
        # # odor = (y + 10)
        # odor[odor < 0] = 0
        # odor[y > self.dist] = 0

        return odor / 10

    # TODO: consolidate rewards
    def get_reward(self, x, y):
        is_done = np.all(np.isclose(self.target, (x, y), atol=self.tolerance))
        # return self.get_odor(x, y), bool(is_done)
        return None, bool(is_done)


# TODO: abstract away and experiment with more complex
#       trail geometries / reward systems
# <codecell>
if __name__ == '__main__':
    from stable_baselines3.common.vec_env import DummyVecEnv

    def env_fn(): return TrailEnv(StraightTrail(), discrete=False)

    env = DummyVecEnv([env_fn for _ in range(8)])
    eval_env = TrailEnv(StraightTrail(), discrete=False)

    # Discrete (untuned)
    # model = PPO("CnnPolicy", env, verbose=1,
    #             n_steps=512,
    #             batch_size=128,
    #             ent_coef=0.01,
    #             gamma=0.98,
    #             gae_lambda=0.9,
    #             use_sde=False,
    #             n_epochs=20,
    #             learning_rate=0.0001,
    #             tensorboard_log='log')

    # TODO: can try tuning with different variations on the CNN:
    # https://github.com/DLR-RM/stable-baselines3/blob/2bb4500948dccba3292135b1e295532fbc32f668/stable_baselines3/common/torch_layers.py#L51
    model = PPO("CnnPolicy", env, verbose=1,
                n_steps=64,
                batch_size=512,
                ent_coef=0.0001,
                gamma=0.995,
                gae_lambda=1.0,
                use_sde=False,
                clip_range=0.1,
                max_grad_norm=2,
                vf_coef=0.715,
                n_epochs=20,
                learning_rate=0.00025,
                tensorboard_log='log',
                # policy_kwargs={
                #     'net_arch': 'medium',
                #     'activation_fn': 'relu'
                # }
                )

    model.learn(total_timesteps=100000, log_interval=5,
                eval_env=eval_env, eval_freq=512)
    model.save('trail_model')
    exit()

# <codecell>
env = TrailEnv(StraightTrail())
model = PPO.load('trail_model')

env.play_anim(model)

# <codecell>
env = TrailEnv(StraightTrail(), discrete=False)
# model = TD3.load('trail_model')
model = PPO.load('trail_model')

obs = env.reset()
plt.imshow(obs)
for _ in range(20):
    # action, _ = model.predict(obs)
    action, _ = model.predict(obs, deterministic=True)
    print(action)
    obs, _, is_done, _ = env.step(action)
    print(is_done)

    # plt.imshow(obs[..., 1])
    plt.imshow(obs)
    plt.show()

    if is_done:
        break

print(env.agent.odor_history)  # TODO: insufficient gradient in odor history

# <codecell>
env = TrailEnv()

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
env = TrailEnv()
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
plt.imshow(obs[..., 1])
print('ODOR', agent.odor_history)

# %%
trail_map = StraightTrail()
trail_map.plot()
plt.show()

# %%
