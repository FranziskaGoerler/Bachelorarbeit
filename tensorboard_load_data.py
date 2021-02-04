from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

folder = './evals/preliminary/action_space/polar/1/'

keys = ['batch/reward_mean', 'eval/reward_sum']

x = EventAccumulator(path=folder)
x.Reload()
x.FirstEventTimestamp()

steps = []
wall_time = []
index = []
count = []
data = []

for k in keys:
    steps.append([e.step for e in x.Scalars(k)])
    wall_time.append([e.wall_time for e in x.Scalars(k)])
    index.append([e.index for e in x.Scalars(k)])
    count.append([e.count for e in x.Scalars(k)])
    data.append([e.value for e in x.Scalars(k)])

plt.subplot(1,2,1)
plt.plot(steps[0], data[0])
plt.subplot(1,2,2)
plt.plot(steps[1], data[1])
plt.show()