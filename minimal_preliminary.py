import gym
import numpy as np
from pathlib import Path

ACTION_SPACE_TYPE = 'scaled'   # 'polar', 'cartesian', 'scaled'
# OBSERVATION_SPACE_TYPE = 'coord+diff'    # 'coordinates', 'coord+diff', 'only diff'
REWARD_FUNCTION = 'distance'    # 'angle', 'distance'
PUNISH_WRONG_DIRECTION = True

class App(gym.Env):
    OK = 0
    TARGET_FOUND = 1
    BOUNDARY_COLLISION = 2
    ROBOT_COLLISION = 3

    def __init__(self, traj_savepath=None):
        self.position = np.array([np.random.rand() * 200 - 100, np.random.rand() * 200 - 100]) # Zufällige x- und y-Werte
        self.start_position = np.array(self.position)
        if 'polar' in ACTION_SPACE_TYPE:
            self.action_space = gym.spaces.box.Box(np.array([-np.pi, 0]), np.array([np.pi, 1]), (2,)) # Winkel, Geschwindigkeit (Radius)
        else:
            self.action_space = gym.spaces.box.Box(np.array([-1, -1]), np.array([1, 1]), (2,))  # x, y
        self.observation_space = gym.spaces.box.Box(np.array([-100, -100]), np.array([100, 100]),(2,))
        self.step_counter = 0
        self.cum_reward = 0
        self.destination = np.array([0., 0.])
        self.done = False

        self.traj_savepath = traj_savepath
        if self.traj_savepath is not None:
            self.init_traj()

    def step(self, action):
        pre_position = np.array(self.position)
        pre_vector_to_target = self.destination - self.position
        pre_distance = np.sqrt(np.sum((pre_vector_to_target) ** 2)) # Euklidische Distanz
        action_length = np.sqrt(np.sum(action ** 2))
        if 'scaled' in ACTION_SPACE_TYPE and 'unscaled' not in ACTION_SPACE_TYPE:
            if action_length > 1:
                action /= action_length
                action_length = 1
        if 'polar' in ACTION_SPACE_TYPE:
            action_length = action[1]
            action_angle = action[0]
            action_cartesian = np.array([action_length * np.cos(action[0]), action_length * np.sin(action[0])])  # x, y Berrechnen
        else:
            action_cartesian = action
            action_angle = np.arctan2(action[1], action[0])
        self.step_counter += 1

        self.position += action_cartesian
        distance = np.sqrt(np.sum((self.destination - self.position) ** 2))

        reward = 0
        signal = App.OK

        if distance <= 5:
            reward = 1000
            # self.position = np.array([100.,100.])
            self.done = True
            signal = App.TARGET_FOUND
            print('Das Ziel wurde erreicht nach {} Schritten '.format(self.step_counter))

        elif not -100 <= self.position[0] <= 100 or not -100 <= self.position[1] <= 100:
            # reward = 0
            self.position = pre_position
            self.done = True
            signal = App.BOUNDARY_COLLISION

        elif self.step_counter >= 200:
            self.done = True

        elif 'distance' in REWARD_FUNCTION:
            delta_distance = pre_distance - distance
            # (1) Belohnung der Distanzverringerung zum Ziel
            if delta_distance > 0 or PUNISH_WRONG_DIRECTION: reward = (5 / np.sqrt(2)) * delta_distance

        elif 'angle' in REWARD_FUNCTION:
            best_angle_to_destination = np.arctan2(pre_vector_to_target[1], pre_vector_to_target[0])  # Wäre der bestmögliche Winkel gewesen
            angle_to_destination = action_angle
            angle_error = best_angle_to_destination - angle_to_destination
            if angle_error > np.pi:
                angle_error -= 2*np.pi
            elif angle_error < -np.pi:
                angle_error += 2*np.pi
            reward_factor = (5 / (np.exp(np.pi / 2) - 1)) * action_length
            abs_err = abs(angle_error)
            # (2) Belohnung abhängig von Verhältnis des gewählten Winkels zum perfekten Winkel
            if abs_err < np.pi / 2:
                reward = (np.exp(-abs_err + np.pi / 2) - 1) * reward_factor
            elif PUNISH_WRONG_DIRECTION:
                reward = - 5 * action_length * ((2 / np.pi) * abs_err - 1)

        # (3) Nur Belohnung, wenn die Bewegungsrichtung geradlinig zum Ziel ist, mit einer Toleranz von 1 Grad
        # elif np.abs(angle_error) < (np.pi / 180):
        #    reward = 5 * action_length

        self.cum_reward += reward

        if self.done:
            movement_vector = self.position - self.start_position
            angle = np.arctan2(movement_vector[1], movement_vector[0]) / np.pi * 180

            print('Die Episode ist beendet. Kumulierte Belohnung: {}, Anzahl Schritte: {}, Bewegungsrichtung: {}, Endposition: ({}, {}) '
                  .format(self.cum_reward, self.step_counter, angle, self.position[0], self.position[1]))

        if self.traj_savepath is not None:
            self.save_traj(signal)

        return (self.position, reward, self.done, dict())

    def reset(self):
        self.position = np.array([np.random.rand() * 200 - 100, np.random.rand() * 200 - 100])
        self.start_position = np.array(self.position)
        self.cum_reward = 0
        self.step_counter = 0
        self.done = False
        return self.position

    def render(self, mode='human'):
        pass

    def init_traj(self):
        self.traj = []
        Path(self.traj_savepath).mkdir(parents=True, exist_ok=True)

        run_number = 0
        for p in Path(self.traj_savepath).iterdir():
            if p.is_file() and p.stem.isnumeric():
                if int(p.stem) > run_number:
                    run_number = int(str(p.stem))
        run_number += 1

        # robpos_savepath = self.traj_savepath + ('/' + str(run_number) + '_robpos.csv')
        # with open(robpos_savepath, 'w') as f:
        #     for r in self.robots:
        #         f.write('{} {} '.format(r.x, r.y))

        self.traj_savepath += ('/' + str(run_number) + '.csv')

    def save_traj(self, signal):
        if not self.done:
            self.traj.append([self.position[0], self.position[1], 0])
        else:
            if signal == App.TARGET_FOUND:
                self.traj.append([self.position[0], self.position[1], 'target'])
            elif signal == App.ROBOT_COLLISION:
                self.traj.append([self.position[0], self.position[1], 'robot'])
            elif signal == App.BOUNDARY_COLLISION:
                self.traj.append([self.position[0], self.position[1], 'boundary'])
            else:
                self.traj.append([self.position[0], self.position[1], 'time'])

            with open(self.traj_savepath, 'a') as f:
                for t in self.traj:
                    f.write('{} {} {}\n'.format(t[0], t[1], t[2]))
            self.traj = []

