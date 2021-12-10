"""Plots of great magnificence and majesty"""

# <codecell>
import torch
from pytorch_grad_cam import GradCAM
from stable_baselines3 import PPO

from trail import TrailEnv
from trail_map import *

# <codecell>
# 3-branch trails
trails = [
    StraightTrail(end=np.array([-15, 15])),
    StraightTrail(end=np.array([0, 15])),
    StraightTrail(end=np.array([15, 15])),
]

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for trail, ax in zip(trails, axs.ravel()):
    im = trail.plot(ax=ax)

fig.colorbar(im)
fig.suptitle('3-branch trails')
fig.tight_layout()
plt.savefig('fig/3_branch.png')

# %% GRAD CAM VIZ: 3-branch - north east
three_branch_photos = [0, 0, 0]

global_discrete = True

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

fig.suptitle('3-branch northeast trail')
fig.tight_layout()

plt.savefig('fig/3_branch_ne.png')

three_branch_photos[-1] = obs

# <codecell>
# 3-branch northwest
trail_map = StraightTrail(end=np.array([-15, 15]), narrow_factor=2)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(2, 6, figsize=(18, 6))
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

    row = (ax_iter + 1) // 6
    col = (ax_iter + 1) % 6
    axs[row, col].remove()
    ax2 = fig.add_subplot(2, 6, ax_iter + 2, projection='polar')

    # ax2 = plt.subplot(122, projection='polar')
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    ax_iter += 2

    if is_done:
        break
    # plt.show()

fig.suptitle('3-branch northwest trail')
fig.tight_layout()

plt.savefig('fig/3_branch_nw.png')

three_branch_photos[0] = obs

# env.map.plot()

# <codecell>
# 3-branch north
trail_map = StraightTrail(end=np.array([0, 15]), narrow_factor=2)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(3, 6, figsize=(18, 9))
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

    row = (ax_iter + 1) // 6
    col = (ax_iter + 1) % 6
    axs[row, col].remove()
    ax2 = fig.add_subplot(3, 6, ax_iter + 2, projection='polar')

    # ax2 = plt.subplot(122, projection='polar')
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    ax_iter += 2

    if is_done:
        break
    # plt.show()

fig.suptitle('3-branch north trail')
fig.tight_layout()

plt.savefig('fig/3_branch_n.png')

three_branch_photos[1] = obs

# <codecell>
# 3 branch photo finish
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
names = ['Northwest', 'North', 'Northeast']

for name, photo, ax in zip(names, three_branch_photos, axs):
    ax.imshow(photo)
    ax.set_title(name)

plt.savefig('fig/three_branch_photo.png')

# <codecell>
# three-branch northeast spotlight
trail_map = StraightTrail(end=np.array([15, 15]), narrow_factor=2)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(4, 4, figsize=(12, 12))
flat_axs = axs.ravel()

obs = env.reset()
for i in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)
    print('position:', env.agent.position)

    # fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # ax1 = plt.subplot(121)
    ax1 = flat_axs[i]
    ax1.imshow(obs)
    ax1.set_title(f't={i+1}')

    if is_done:
        break
    # plt.show()

fig.suptitle('3-branch northeast trail')
fig.tight_layout()

plt.savefig('fig/3_branch_ne_spotlight.png')

# <codecell>
# 3-branch north east closeup
trail_map = StraightTrail(end=np.array([15, 15]), narrow_factor=2)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(4, 2, figsize=(8, 16))
flat_axs = axs.ravel()
ax_iter = 0

obs = env.reset()
for i in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)
    print('position:', env.agent.position)

    # fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # ax1 = plt.subplot(121)
    if (i + 1) in [2, 3, 4, 5]:
        ax1 = flat_axs[ax_iter]
        ax1.imshow(obs)

        obs_t, _ = pi.obs_to_tensor(obs)
        grayscale_cam = cam(input_tensor=obs_t, target_category=None)
        ax1.imshow(grayscale_cam[0], alpha=0.5, cmap='bone')

        row = (ax_iter + 1) // 2
        col = (ax_iter + 1) % 2
        axs[row, col].remove()
        ax2 = fig.add_subplot(4, 2, ax_iter + 2, projection='polar')

        # ax2 = plt.subplot(122, projection='polar')
        value = plot_probs(obs, ax2)

        ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

        ax_iter += 2

    if is_done:
        break
    # plt.show()

fig.suptitle('3-branch northeast close-up')
fig.tight_layout()

plt.savefig('fig/3_branch_ne_closeup.png')

# <codecell>
# 5-branch trails
trails = [
    StraightTrail(end=np.array([-15, 0])),
    StraightTrail(end=np.array([-15, 15])),
    StraightTrail(end=np.array([0, 15])),
    StraightTrail(end=np.array([15, 15])),
    StraightTrail(end=np.array([15, 0])),
]

fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for trail, ax in zip(trails, axs.ravel()):
    im = trail.plot(ax=ax)

fig.colorbar(im)
fig.suptitle('5-branch trails')
fig.tight_layout()
plt.savefig('fig/5_branch.png')

# %%

# <codecell>
# 5-branch west
five_branch_photos = [0, 0, 0, 0, 0]

global_discrete = True

model = PPO.load('trained/branch_5.zip')
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


trail_map = StraightTrail(end=np.array([-15, 0]), narrow_factor=1)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


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


fig.suptitle('5-branch west trail')
fig.tight_layout()

plt.savefig('fig/5_branch_w.png')

five_branch_photos[0] = obs

# <codecell>
# 5-branch northwest

trail_map = StraightTrail(end=np.array([-15, 15]), narrow_factor=1)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(2, 6, figsize=(18, 6))
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

    row = (ax_iter + 1) // 6
    col = (ax_iter + 1) % 6
    axs[row, col].remove()
    ax2 = fig.add_subplot(2, 6, ax_iter + 2, projection='polar')

    # ax2 = plt.subplot(122, projection='polar')
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    ax_iter += 2

    if is_done:
        break
    # plt.show()


fig.suptitle('5-branch northwest trail')
fig.tight_layout()

plt.savefig('fig/5_branch_nw.png')

five_branch_photos[1] = obs

# %%
# 5-branch north

trail_map = StraightTrail(end=np.array([0, 15]), narrow_factor=1)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(3, 6, figsize=(18, 9))
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

    row = (ax_iter + 1) // 6
    col = (ax_iter + 1) % 6
    axs[row, col].remove()
    ax2 = fig.add_subplot(3, 6, ax_iter + 2, projection='polar')

    # ax2 = plt.subplot(122, projection='polar')
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    ax_iter += 2

    if is_done:
        break
    # plt.show()


fig.suptitle('5-branch north trail')
fig.tight_layout()

plt.savefig('fig/5_branch_n.png')

five_branch_photos[2] = obs

# %%
# 5-branch northeast

trail_map = StraightTrail(end=np.array([15, 15]), narrow_factor=1)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


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


fig.suptitle('5-branch northeast trail')
fig.tight_layout()

plt.savefig('fig/5_branch_ne.png')

five_branch_photos[3] = obs

# <codecell>
# 5-branch east
trail_map = StraightTrail(end=np.array([15, 0]), narrow_factor=1)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(3, 8, figsize=(24, 9))
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
    ax2 = fig.add_subplot(3, 8, ax_iter + 2, projection='polar')

    # ax2 = plt.subplot(122, projection='polar')
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    ax_iter += 2

    if is_done:
        break
    # plt.show()


fig.suptitle('5-branch east trail')
fig.tight_layout()

plt.savefig('fig/5_branch_e.png')

five_branch_photos[4] = obs

# <codecell>
# 3 branch photo finish
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
names = ['West', 'Northwest', 'North', 'Northeast', 'East']

for name, photo, ax in zip(names, five_branch_photos, axs):
    ax.imshow(photo)
    ax.set_title(name)

plt.savefig('fig/five_branch_photo.png')

# <codecell>
# five-branch west spotlight
trail_map = StraightTrail(end=np.array([-15, 0]), narrow_factor=1)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(4, 4, figsize=(12, 12))
flat_axs = axs.ravel()

obs = env.reset()
for i in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)
    print('position:', env.agent.position)

    # fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # ax1 = plt.subplot(121)
    ax1 = flat_axs[i]
    ax1.imshow(obs)
    ax1.set_title(f't={i+1}')

    if is_done:
        break
    # plt.show()

fig.suptitle('5-branch west trail')
fig.tight_layout()

plt.savefig('fig/5_branch_w_spotlight.png')

# <codecell>
# 5-branch west closeup
trail_map = StraightTrail(end=np.array([-15, 0]), narrow_factor=1)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)


fig, axs = plt.subplots(4, 2, figsize=(8, 16))
flat_axs = axs.ravel()
ax_iter = 0

obs = env.reset()
for i in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    print(action)
    print(is_done)
    print('last reward:', reward)
    print('position:', env.agent.position)

    # fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # ax1 = plt.subplot(121)
    if (i + 1) in [7, 8, 9, 10]:
        ax1 = flat_axs[ax_iter]
        ax1.imshow(obs)

        obs_t, _ = pi.obs_to_tensor(obs)
        grayscale_cam = cam(input_tensor=obs_t, target_category=None)
        ax1.imshow(grayscale_cam[0], alpha=0.5, cmap='bone')

        row = (ax_iter + 1) // 2
        col = (ax_iter + 1) % 2
        axs[row, col].remove()
        ax2 = fig.add_subplot(4, 2, ax_iter + 2, projection='polar')

        # ax2 = plt.subplot(122, projection='polar')
        value = plot_probs(obs, ax2)

        ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

        ax_iter += 2

    if is_done:
        break
    # plt.show()

fig.suptitle('5-branch west close-up')
fig.tight_layout()

plt.savefig('fig/5_branch_w_closeup.png')

# %%
