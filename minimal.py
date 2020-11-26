import gym
import numpy as np

N_BINS = 8
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

class App(gym.Env):
    def __init__(self):
        self.position = np.array([np.random.rand() * SCREEN_WIDTH, np.random.rand() * SCREEN_HEIGHT]) # Zufällige x- und y-Werte
        # self.position = np.array([0., 0.])
        self.start_position = np.array(self.position)
        self.action_space = gym.spaces.box.Box(np.array([-3, -3]), np.array([3, 3]), (2,))
        self.observation_space = gym.spaces.box.Box(np.array([0, 0, -800, -800]), np.array([800, 800, 800, 800]),(4,)) # Ziel und Distanzvektor
        self.step_counter = 0
        self.cum_reward = 0
        self.start_targets = []
        self.end_targets = []

        space = (SCREEN_HEIGHT - 100) / (N_BINS - 1)
        for i in range(N_BINS):
            self.start_targets.append([50, 50 + i * space])
            self.start_targets.append([SCREEN_WIDTH - 50, 50 + i * space])

        self.destination = np.array(self.start_targets[np.random.randint(8)])

    def step(self, action):
        pre_position = np.array(self.position)
        pre_vector_to_target = self.destination - self.position
        pre_distance = np.sqrt(np.sum((pre_vector_to_target) ** 2)) # Euklidische Distanz
        action_length = np.sqrt(np.sum(action ** 2))
        self.step_counter += 1

        # Normierung auf Länge 1, damit der Agent nicht immer nur diagonal (45 Grad) läuft, um maximale Belohnung zu erhalten
        if action_length > 3:
            action /= action_length
            action_length = 3

        self.position += action
        done = False
        vector_to_target = self.destination - self.position
        distance = np.sqrt(np.sum((vector_to_target) ** 2))
        best_angle_to_destination = np.arctan2(pre_vector_to_target[1], pre_vector_to_target[0]) # Wäre der bestmögliche Winkel gewesen
        angle_to_destination = np.arctan2(action[1], action[0])
        angle_error = best_angle_to_destination - angle_to_destination
        delta_distance = pre_distance - distance
        reward_factor = (1 / (np.exp(np.pi / 2) - 1)) * action_length
        reward = 0

        if distance <= 1:
            reward = 5000
            done = True
            print('Das Ziel wurde erreicht nach {} Schritten '.format(self.step_counter))

        elif not 0 <= self.position[0] <= 800 or not 0 <= self.position[1] <= 800:
            # reward = -5
            self.position = pre_position
            done = True

        elif self.step_counter >= 500:
            done = True

        # (1) Belohnung der Distanzverringerung zum Ziel
        # elif delta_distance > 0:
        # else: reward = (5 / np.sqrt(2)) * delta_distance
        # else: reward = 0

        # (2) Belohnung abhängig vom Verhältnis des gewählten Winkels zum perfekten Winkel
        elif angle_error <= 0 and angle_error >= - np.pi / 2:
            reward = (np.exp(angle_error + np.pi / 2) - 1) * reward_factor
        elif angle_error > 0 and angle_error <= np.pi / 2:
            reward = (np.exp(- angle_error + np.pi / 2) - 1) * reward_factor
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

            print('Die Episode ist beendet. Kummulierte Belohnung: {}, Anzahl Schritte: {}, Bewegungsrichtung: {}, Endposition: ({}, {}), Zielposition: ({}, {}) '
                  .format(self.cum_reward, self.step_counter, angle, self.position[0], self.position[1], self.destination[0], self.destination[1]))

        return ([self.position[0], self.position[1], vector_to_target[0], vector_to_target[1]], reward, done, dict())

    def reset(self):
        self.destination = np.array(self.start_targets[np.random.randint(8)])
        self.position = np.array([np.random.rand() * SCREEN_WIDTH, np.random.rand() * SCREEN_HEIGHT])
        self.start_position = np.array(self.position)
        self.cum_reward = 0
        self.step_counter = 0
        vector_to_target = self.destination - self.position
        return [self.position[0], self.position[1], vector_to_target[0], vector_to_target[1]]

    def render(self, mode='human'):
        pass

