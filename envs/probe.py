"""
Testing ground for probing a trained agent
"""
# <codecell>
import torch
from matplotlib.animation import FuncAnimation
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from stable_baselines3 import PPO

from trail import TrailEnv
from trail_map import *


# <codecell>
global_discrete = True
global_treadmill = True
trail_class = MeanderTrail
# trail_args = {'width': 3, 'length': 75, 'radius': 100, 'diff_rate': 0.04, 'breaks': [(0.5, 0.8)]}
trail_args = {'width': 3, 'length': 69, 'radius': 100, 'diff_rate': 0.04}


# <codecell>
# RUNNING AGENT ON TRAILS

# trail_map = StraightTrail(end=np.array([15, 15]), narrow_factor=2)
trail_map = trail_class(**trail_args, heading=-np.pi/4)
env = TrailEnv(trail_map, discrete=global_discrete, treadmill=global_treadmill)
plt.show()

model = PPO.load('trail_model.zip', device='cpu')
# model = PPO.load('trained/narrow5_mixed.zip', device='cpu')

model.policy = model.policy.to('cpu')

obs = env.reset()
plt.imshow(obs)
for _ in range(100):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)

    plt.imshow(obs)
    plt.show()

    if is_done:
        break

env.map.plot(ax=plt.gca())
plt.plot(*zip(*env.agent.position_history), linewidth=2, color='black')
plt.savefig('out.png')

print(env.agent.odor_history)

# <codecell>
# FUNC ANIMATION

# env = TrailEnv(StraightTrail(end=np.array([-15, 0]), narrow_factor=1), discrete=global_discrete)
# model = PPO.load('trained/branch_9_close2.zip', device='cpu')
trail_map = trail_class(**trail_args)
env = TrailEnv(trail_map, discrete=global_discrete, treadmill=global_treadmill)
model = PPO.load('trail_model.zip', device='cpu')

obs = env.reset()
frames = [obs]
plt.imshow(obs)
for _ in range(100):
    action, _ = model.predict(obs, deterministic=False)
    obs, _, is_done, _ = env.step(action)
    frames.append(obs)

    if is_done:
        print('reach end')
        break

ani = FuncAnimation(plt.gcf(), lambda f: plt.imshow(f), frames=frames)
ani.save('out.gif')

# <codecell>
# VIEWING ACTOR NETWORK DECISIONS

'''
Relevant NN's:
- features_extractor
- action_net
- value_net
'''
model.policy.to('cpu')

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

# <codecell>
# MAKE OBS PLOT
plt.imshow(obs)
plt.title('Example observation')
plt.savefig('fig/example.png')

# %% GRAD CAM VIZ
model = PPO.load('trail_model.zip')
model.policy.to('cpu')
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


trail_map = StraightTrail(end=np.array([-15, 0]), narrow_factor=2)
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
