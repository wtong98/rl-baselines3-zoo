"""
Testing ground for probing a trained agent
"""
# <codecell>
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from stable_baselines3 import PPO

from trail import TrailEnv
from trail_map import *


# <codecell>
global_discrete = True
trail_class = RandomStraightTrail
trail_args = {'narrow_factor': 1}


# <codecell>
# RUNNING AGENT ON TRAILS

trail_map = StraightTrail(end=np.array([0, 15]), narrow_factor=3)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)

# model = PPO.load('trail_model.zip')
model = PPO.load('trained/branch_3.zip')

obs = env.reset()
plt.imshow(obs)
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)

    plt.imshow(obs)
    plt.show()

    if is_done:
        break

env.map.plot()

print(env.agent.odor_history)

# <codecell>
# VIEWING ACTOR NETWORK DECISIONS

'''
Relevant NN's:
- features_extractor
- action_net
- value_net
'''

pi = model.policy
all_actions = torch.arange(model.action_space.n)
thetas = - np.arange(8) / 8 * 2 * np.pi + np.pi / 2


@torch.no_grad()
def plot_probs(obs, ax):
    obs_t, _ = pi.obs_to_tensor(obs)
    value, probs, _ = pi.evaluate_actions(obs_t, all_actions)

    ax.bar(thetas, np.exp(probs), alpha=0.7)
    ax.set_yticks(np.arange(1, step=0.25))
    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
    return value


trail_map = StraightTrail(end=np.array([0, 15]), narrow_factor=2)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)

model = PPO.load('trained/branch_5.zip')

obs = env.reset()
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)
    print('position:', env.agent.position)

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    ax1 = plt.subplot(121)
    ax1.imshow(obs)

    ax2 = plt.subplot(122, projection='polar')
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    plt.show()

    if is_done:
        break

env.map.plot()

# %% GRAD CAM VIZ
model = PPO.load('trained/branch_3.zip')
pi = model.policy
all_actions = torch.arange(model.action_space.n)
thetas = - np.arange(8) / 8 * 2 * np.pi + np.pi / 2


class VisualPolicy(torch.nn.Module):
    def __init__(self, pi):
        super().__init__()
        self.pi = pi
        self.all_actions = torch.arange(8)

    def forward(self, obs):
        _, probs, _ = self.pi.evaluate_actions(obs, self.all_actions)
        return probs.unsqueeze(0)


cam = GradCAM(
    model=VisualPolicy(pi),
    target_layers=[pi.features_extractor.cnn[5]]
)


@torch.no_grad()
def plot_probs(obs, ax):
    obs_t, _ = pi.obs_to_tensor(obs)
    value, probs, _ = pi.evaluate_actions(obs_t, all_actions)

    ax.bar(thetas, np.exp(probs), alpha=0.7)
    ax.set_yticks(np.arange(1, step=0.25))
    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
    return value


trail_map = StraightTrail(end=np.array([15, 15]), narrow_factor=2)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


obs = env.reset()
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)
    print('position:', env.agent.position)

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    ax1 = plt.subplot(121)
    ax1.imshow(obs)

    obs_t, _ = pi.obs_to_tensor(obs)
    grayscale_cam = cam(input_tensor=obs_t, target_category=None)
    ax1.imshow(grayscale_cam[0], alpha=0.5, cmap='bone')

    ax2 = plt.subplot(122, projection='polar')
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    plt.show()

    if is_done:
        break

env.map.plot()

# %%
