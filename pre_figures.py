import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pdb

PPO = False
RM = 5000    # running mean for TD3 training reward
TD3_eval_scaling = 0.2          # 0.2 for comparison with PPO

# parent_folder = 'preliminary_minimal/action_space2'
# parent_folder = 'preliminary_minimal/reward_function2'
# parent_folder = 'preliminary/observation_space_angle'
parent_folder = 'preliminary/algorithm_angle'
# folder_names = ['polar', 'cartesian', 'scaled']
folder_names = ['td3', 'ppo']
# folder_names = ['coord+diff', 'coordinates', 'only diff']
# folder_names = ['angle-False', 'angle-True', 'distance-False', 'distance-True']

def tensorboard_loaddata(dir, ppo=PPO):
    if ppo:
        keys = ['batch/reward_mean', 'eval/reward_sum']   # ppo
    else:
        keys = ['reward/batch', 'eval/reward_sum']   # td3

    x = EventAccumulator(path=dir)
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
        # index.append([e.index for e in x.Scalars(k)])
        # count.append([e.count for e in x.Scalars(k)])
        data.append([e.value for e in x.Scalars(k)])
    return wall_time, steps, data

if __name__ == '__main__':
    y_train_all = []
    y_mean_all = []
    x_train_all = []
    x_mean_all = []

    for folder in folder_names:

        y_train_all.append([])
        y_mean_all.append([])
        x_train_all.append([])
        x_mean_all.append([])

        for j in range(3):    # 0, 2, 4

            if folder == 'td3':
                j = 0

            dir = './evals/{}/{}/{}/'.format(parent_folder, folder, j + 1)
            try:
                if 'ppo' in folder:
                    wall_time, steps, data = tensorboard_loaddata(dir, ppo=True)
                else:
                    wall_time, steps, data = tensorboard_loaddata(dir)
            except:
                print('Cannot load data from {}'.format(dir))
                continue

            time = wall_time[1]
            x = steps[1]
            y = data[1]
            x_train = steps[0]
            y_train = data[0]

            y_mean = [[]]
            t_prev = time[0]

            for i in range(len(time)):
                t = time[i]
                if t - t_prev < 1.:
                    y_mean[-1].append(y[i])
                else:
                    y_mean.append([])
                    y_mean[-1].append(y[i])

                t_prev = t

            y_mean = [np.mean(y) for y in y_mean]

            x_mean = np.arange(len(y_mean))

            if not PPO and not 'ppo' in folder:
                x_mean = x_mean*TD3_eval_scaling   # PPO evaluates every 50 steps, TD3 every 10

                rm = RM   # running mean
                y_train_rm = []
                x_train_rm = []
                y_temp = []
                x_ref = x_train[0]
                x_last = x_train[0]
                for x, y in zip(x_train, y_train):
                    if x - x_ref < rm:
                        y_temp.append(y)
                    else:
                        y_train_rm.append(np.mean(y_temp))
                        y_temp = []
                        x_train_rm.append(x_last)
                        x_ref = x
                    x_last = x

                y_train = y_train_rm
                x_train = x_train_rm

                # y_train = [np.mean(y_train[rm*i:rm*i+rm]) for i in range(len(y_train)//rm)]
                # x_train = [x_train[rm*i+rm-1] for i in range(len(x_train)//rm)]

            y_train_all[-1].append(y_train)
            y_mean_all[-1].append(y_mean)
            x_train_all[-1].append(x_train)
            x_mean_all[-1].append(x_mean)

    # min_length = min([len(y) for y_t in y_train_all for y in y_t])

    y_train_avg = []
    x_train_avg = []

    for x_t, y_t in zip(x_train_all, y_train_all):
        min_length = min([len(y) for y in y_t])
        y_t2 = [y[:min_length] for y in y_t]
        y_t2 = np.array(y_t2)
        y_t2 = np.mean(y_t2, axis=0)
        y_train_avg.append(y_t2)
        x_train_avg.append(x_t[0][:min_length])

    # x_train_all = x_train[:min_length]


    # min_length = min([len(y) for y_t in y_mean_all for y in y_t])

    y_mean_avg = []
    x_mean_avg = []

    for x_m, y_m in zip(x_mean_all, y_mean_all):
        min_length = min([len(y) for y in y_m])
        y_m2 = [y[:min_length] for y in y_m]
        y_m2 = np.array(y_m2)
        y_m2 = np.mean(y_m2, axis=0)
        y_mean_avg.append(y_m2)
        x_mean_avg.append(x_m[0][:min_length])

    # x_mean_all = x_mean[:min_length]


    plt.subplot(1, 2, 1)
    for x_t, y_t in zip(x_train_avg, y_train_avg):
        l, = plt.plot(x_t, y_t)
    plt.title('Belohnung: Training')
    plt.xlabel('Schritt')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,3))
    plt.ylabel('Belohnung pro Schritt')
    plt.subplot(1, 2, 2)
    lines = []

    for x_m, y_m in zip(x_mean_avg, y_mean_avg):
        l, = plt.plot(x_m, y_m)
        lines.append(l)
    plt.title('Belohnung: Evaluierung')
    plt.xlabel('Evaluierung Nr.')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,3))
    plt.ylabel('Belohnung pro Episode')
    plt.figlegend(lines, folder_names)
    plt.show()