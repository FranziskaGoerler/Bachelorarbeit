import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
N_BINS = 4

BOT_RADIUS = 17.5
# BOT_RADIUS = 7.5
TARGET_WIDTH = 35
# TARGET_WIDTH = 15
TARGET_IN_CENTER = False
# TARGET_IN_CENTER = True

# LOAD_PATH = './evals/action_space/polar/3/'
# LOAD_PATH = './evals/observation_space/only diff/3/'
# LOAD_PATH = './evals/reward_function/angle-False/3/'
LOAD_PATH = './evals/preliminary/algortihm/td3/1/'

load_number = 1

traj_file = LOAD_PATH + str(load_number) + '.csv'
robots_file = LOAD_PATH + str(load_number) + '_robpos.csv'

with open(traj_file, 'r') as f:
    traj = f.readlines()

x_pos = [[]]
y_pos = [[]]
endstate = []

for t in traj:
    t_split = t.split()
    x_pos[-1].append(float(t_split[0]))
    y_pos[-1].append(float(t_split[1]))

    if t_split[2] != '0':
        endstate.append(t_split[2])
        x_pos.append([])
        y_pos.append([])

robs = True
try:
    with open(robots_file, 'r') as f:
        robpos = f.read().split()
except:
    robs = False

robots = []
if robs:
    for i in range(len(robpos)//2):
        robots.append([robpos[2*i],robpos[2*i+1]])

plt.figure(figsize=(8,8))
plt.tight_layout()
ax = plt.gca()
ax.axis([0, SCREEN_WIDTH, 0, SCREEN_HEIGHT])
# ax.axis([-SCREEN_WIDTH/2, SCREEN_WIDTH/2, -SCREEN_HEIGHT/2, SCREEN_HEIGHT/2])
plt.tick_params(
    # axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    right=False,
    labelleft=False
    )

patches = []
space = (SCREEN_HEIGHT - 100) / (N_BINS - 1)


if TARGET_IN_CENTER:
    patches.append(Rectangle((0- TARGET_WIDTH / 2, 0 - TARGET_WIDTH / 2), TARGET_WIDTH, TARGET_WIDTH, color='m'))
    ax.add_patch(patches[-1])
else:
    for i in range(N_BINS):
        patches.append(Rectangle((50 -TARGET_WIDTH/2, 50 -TARGET_WIDTH/2 + i * space), TARGET_WIDTH, TARGET_WIDTH, color='m'))
        patches.append(Rectangle((SCREEN_WIDTH - 50 -TARGET_WIDTH/2, 50 -TARGET_WIDTH/2 + i * space), TARGET_WIDTH, TARGET_WIDTH, color='m'))
        ax.add_patch(patches[-2])
        ax.add_patch(patches[-1])

for r in robots:
    patches.append(Circle((r[0], r[1]), BOT_RADIUS, color='k'))
    ax.add_patch(patches[-1])

for x, y, state in zip(x_pos, y_pos, endstate):
    c = 'k'
    if state == 'robot' or state == 'boundary':
        c = 'r'
    elif state == 'target':
        c = 'g'
    elif state == 'time':
        c = 'b'

    ax.plot(x, y, color=c)
    ax.plot(x[0], y[0], marker='o', color =c, markersize=8, fillstyle='none', clip_on=False)
    ax.plot(x[-1], y[-1], marker='o', color =c, markersize=8, fillstyle='full', clip_on=False)


# pc = PatchCollection(patches)
# ax = plt.gca()
#
# # Add collection to axes
# ax.add_collection(pc)


plt.savefig(LOAD_PATH + str(load_number) + '_traj.pdf')
plt.show()
