"""Making plots of the branch 8 case

author: William Tong (wtong@g.harvard.edu)
date: 1/23/2022
"""

# <codecell>
from matplotlib.animation import FuncAnimation
import torch
from pytorch_grad_cam import GradCAM
from stable_baselines3 import PPO

import sys
sys.path.append('../')

from trail import TrailEnv
from trail_map import *

# <codecell>
global_discrete = True
global_treadmill = False

model = PPO.load('trained/old/branches/branch_8.zip', device='cpu')
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


trail_map = StraightTrail(end=np.array([-15, 15]), narrow_factor=2)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete, treadmill=global_treadmill)


fig, axs = plt.subplots(4, 8, figsize=(24, 12))
flat_axs = axs.ravel()
ax_iter = 0

obs = env.reset()
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)
    print('position:', env.agent.position)

    # fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # ax1 = plt.subplot(121)
    ax1 = flat_axs[ax_iter]
    ax1.imshow(obs)

    obs_t, _ = pi.obs_to_tensor(obs)
    grayscale_cam = cam(input_tensor=obs_t, target_category=None)
    ax1.imshow(grayscale_cam[0], alpha=0.5, cmap='bone')

    row = (ax_iter + 1) // 8
    col = (ax_iter + 1) % 8
    axs[row, col].remove()
    ax2 = fig.add_subplot(4, 8, ax_iter + 2, projection='polar')

    # ax2 = plt.subplot(122, projection='polar')
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    ax_iter += 2

    if is_done:
        break
    # plt.show()

fig.suptitle('8-branch northwast trail')
fig.tight_layout()

plt.savefig('fig/branch_8/nw.png')

# <codecell>
# FUNC ANIMATION

obs = env.reset()
frames = [obs]
plt.imshow(obs)
for _ in range(20):
    action, _ = model.predict(obs, deterministic=False)
    obs, _, is_done, _ = env.step(action)
    frames.append(obs)

    if is_done:
        print('reach end')
        break

ani = FuncAnimation(plt.gcf(), lambda f: plt.imshow(f), frames=frames)
ani.save('fig/branch_8/nw_anim.gif')
# %%
