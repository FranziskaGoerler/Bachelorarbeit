import gym
import numpy as np

class App(gym.Env):
    def __init__(self):
        # self.position = np.array([np.random.rand() * 200 - 100, np.random.rand() * 200 - 100]) # Zufällige x- und y-Werte
        self.position = np.array([0., 0.])
        self.start_position = np.array(self.position)
        self.action_space = gym.spaces.box.Box(np.array([-1, -1]), np.array([1, 1]), (2,))
        self.observation_space = gym.spaces.box.Box(np.array([-100, -100]), np.array([100, 100]),(2,))
        self.step_counter = 0
        self.cum_reward = 0
        self.destination = np.array([30., -70.])

    def step(self, action):
        pre_position = np.array(self.position)
        pre_vector_to_target = self.destination - self.position
        pre_distance = np.sqrt(np.sum((pre_vector_to_target) ** 2)) # Euklidische Distanz
        action_length = np.sqrt(np.sum(action ** 2))
        self.step_counter += 1

        # Normierung auf Länge 1, damit der Agent nicht immer nur diagonal (45 Grad) läuft, um maximale Belohnung zu erhalten
        if action_length > 1:
            action /= action_length
            action_length = 1

        self.position += action
        done = False
        distance = np.sqrt(np.sum((self.destination - self.position) ** 2))
        best_angle_to_destination = np.arctan2(pre_vector_to_target[1], pre_vector_to_target[0]) # Wäre der bestmögliche Winkel gewesen
        angle_to_destination = np.arctan2(action[1], action[0])
        angle_error = best_angle_to_destination - angle_to_destination
        delta_distance = pre_distance - distance
        reward_factor = (5 / (np.exp(np.pi / 2) - 1)) * action_length
        reward = 0

        if distance <= 1:
            reward = 10000
            # self.position = np.array([100.,100.])
            done = True
            print('Das Ziel wurde erreicht nach {} Schritten '.format(self.step_counter))

        elif not -100 <= self.position[0] <= 100 or not -100 <= self.position[1] <= 100:
            # reward = -5
            self.position = pre_position
            done = True

        elif self.step_counter >= 200:
            done = True

        # (1) Belohnung der Distanzverringerung zum Ziel
        elif delta_distance > 0:
            reward = (5 / np.sqrt(2)) * delta_distance
        else: reward = 0

        # (2) Belohnung abhängig von Verhältnis des gewählten Winkels zum perfekten Winkel
        # elif angle_error <= 0 and angle_error >= - np.pi / 2:
        #     reward = (np.exp(angle_error + np.pi / 2) - 1) * reward_factor
        # elif angle_error > 0 and angle_error <= np.pi / 2:
        #     reward = (np.exp(- angle_error + np.pi / 2) - 1) * reward_factor
        # else: # Bestrafung, bei Entfernung vom Ziel
        #     neg_angle_error = np.pi - abs(angle_error)
        #     reward = - (np.exp(- neg_angle_error + np.pi / 2) - 1) * reward_factor

        # (3) Nur Belohnung, wenn die Bewegungsrichtung geradlinig zum Ziel ist, mit einer Toleranz von 45 Grad
        # elif np.abs(angle_error) < (np.pi / 180):
        #    reward = 5 * action_length

        self.cum_reward += reward

        if done == True :
            movement_vector = self.position - self.start_position
            angle = np.arctan2(movement_vector[1], movement_vector[0]) / np.pi * 180

            print('Die Episode ist beendet. Kummulierte Belohnung: {}, Anzahl Schritte: {}, Bewegungsrichtung: {}, Endposition: ({}, {}) '
                  .format(self.cum_reward, self.step_counter, angle, self.position[0], self.position[1]))

        return (self.position, reward, done, dict())

    def reset(self):
        # self.position = np.array([np.random.rand() * 200 - 100, np.random.rand() * 200 - 100])
        self.position = np.array([0., 0.])
        self.start_position = np.array(self.position)
        self.cum_reward = 0
        self.step_counter = 0
        return self.position

    def render(self, mode='human'):
        pass

