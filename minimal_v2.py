import gym
import numpy as np

class App(gym.Env):
    def __init__(self):
        self.position = np.array([0.,0.])
        self.action_space = gym.spaces.box.Box(np.array([-1, -1]), np.array([1, 1]), (2,))
        self.observation_space = gym.spaces.box.Box(np.array([-100.]*4), np.array([100.]*4),(4,))
        self.step_counter = 0
        self.cum_reward = 0
        self.destination = np.array([100., 100.])


    def step(self, action):
        pre_position = np.array(self.position)
        pre_distance = np.sqrt(np.sum((self.destination - self.position) ** 2)) # Euklidische Distanz

        self.step_counter += 1
        # speed = action[0]

        action_length = np.sqrt(np.sum(action**2))
        if action_length > 1:
            action /= action_length
        self.position += action

        done = False


        distance = np.sqrt(np.sum((self.destination - self.position) ** 2))
        speed = pre_distance - distance

        reward = (5 / np.sqrt(2)) * speed
        # else : reward = 0

        if distance <= 1 :
            reward = 1000
            self.position = self.destination
            done = True
            print('Das Ziel wurde erreicht nach {} Schritten '.format(self.step_counter))
        elif not -100 <= self.position[0] < 100 or not -100 <= self.position[1] < 100 : # Wenn Agent des Aktionsraum verlässt
            reward = -100
            self.position = pre_position
            done = True
        elif self.step_counter >= 200:
            done = True

        self.cum_reward += reward

        if done == True :
            print('Die Episode ist beendet. Kummulierte Belohnung: {}, Anzahl Schritte: {}, Position: ({},{}), Zielposition:({}, {})'.format(self.cum_reward, self.step_counter, self.position[0], self.position[1], self.destination[0], self.destination[1]))


        return (list(self.position) + list(self.destination), reward, done, dict())


    def reset(self):
        self.position = np.array([0.,0.])
        self.cum_reward = 0
        self.step_counter = 0


        if np.random.rand() > 0.5:
            self.destination = np.array([100., 100.])
        else:
            self.destination = np.array([-100., -100.])

        return list(self.position) + list(self.destination)

    def render(self, mode='human'):
        pass
