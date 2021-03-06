import numpy as np
import matplotlib.pyplot as plt
from multiple_formatter import multiple_formatter

step_len = 1

pi = np.pi
x = np.arange(1, 101) * (pi/50) - pi
y = np.zeros(100)

reward_factor = (1 / (np.exp(np.pi / 2) - 1)) * step_len
# reward_factor = 1

y[:25] = - step_len * ((2 / np.pi) * np.abs(x[:25]) - 1)
y[75:] = - step_len * ((2 / np.pi) * np.abs(x[75:]) - 1)
y[25:75] = (np.exp(-np.abs(x[25:75]) + np.pi / 2) - 1) * reward_factor

#plt.plot(x,y)
plt.plot(x[24:75],y[24:75], color='C9')
plt.plot(x[:25],y[:25], color='C0')
plt.plot(x[74:],y[74:], color='C0')
plt.ylabel('Belohnung')
plt.xlabel(r'Winkelabweichung $\delta$')

ax = plt.gca()
ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=1)
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

plt.show()