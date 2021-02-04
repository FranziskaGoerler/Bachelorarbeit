import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from multiple_formatter import multiple_formatter
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
N_BINS = 4

BOT_RADIUS = 17.5
TARGET_WIDTH = 35

ROB_R = 80
C = 60

def gauss_func(x, mean=0, std=1):
    return np.exp(-((x-mean)/std)**2 / 2) / (std * np.sqrt(2*np.pi))

def add_gauss(mean, std, height=1., color=None):
    # y = norm(mean, std).pdf(x)
    y = gauss_func(x, mean,std)
    print(max(y))
    y_scaled = height * y / max(y)
    if color is None:
        plt.plot(x, y_scaled)
    else:
        plt.plot(x, y_scaled, color=color)
    return y_scaled

pi = np.pi
x = np.arange(101) * (pi/100) - pi/2

plt.subplot(1,2,1)
g0 = add_gauss(0, pi/4, 1, 'r')

distance_scaling = (-2 / np.exp(800/C)) * np.exp((800 - (ROB_R - 2*BOT_RADIUS))/C)

g1 = add_gauss(2*pi/12, pi/10, distance_scaling, 'g')
plt.plot(x, g0+g1, color='b')
plt.legend(['target', 'obstacle', 'sum'])
ax = plt.gca()
ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=1)
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))


patches = []
space = (SCREEN_HEIGHT - 100) / (N_BINS - 1)
plt.subplot(1,2,2)
ax = plt.gca()
ax.axis([0, SCREEN_WIDTH, 0, SCREEN_HEIGHT])
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
for i in range(N_BINS):
    patches.append(Rectangle((50 - TARGET_WIDTH / 2, 50 - TARGET_WIDTH / 2 + i * space), TARGET_WIDTH, TARGET_WIDTH, color='m'))
    if i == 1:
        patches.append(Rectangle((SCREEN_WIDTH - 50 - TARGET_WIDTH / 2, 50 - TARGET_WIDTH / 2 + i * space), TARGET_WIDTH, TARGET_WIDTH, color='r'))
    else:
        patches.append(Rectangle((SCREEN_WIDTH - 50 - TARGET_WIDTH / 2, 50 - TARGET_WIDTH / 2 + i * space), TARGET_WIDTH, TARGET_WIDTH, color='m'))
    ax.add_patch(patches[-2])
    ax.add_patch(patches[-1])

patches.append(Circle((400, 283), BOT_RADIUS, color='b'))
ax.add_patch(patches[-1])

rob_angle = -np.pi/6
rob_r = ROB_R
rob_x = 400 + rob_r*np.cos(rob_angle)
rob_y = 283 + rob_r*np.sin(rob_angle)
patches.append(Circle((rob_x, rob_y), BOT_RADIUS, color='g'))
ax.add_patch(patches[-1])


# plt.figure()
# x = np.arange(800)
# # y = (80 - x/10)**3
# C = 40
# y = -(2 / np.exp(800/C)) * np.exp((800-x)/C)
# # print(np.exp(8))
# plt.plot(x,y)

plt.show()