import gym
import numpy as np

class App(gym.Env):
    def __init__(self):
        self.position = 0
        self.action_space = gym.spaces.box.Box(np.array([-1]), np.array([1]), (1,)) # Aktionen zwischen -1 (mit max. Geschwindigkeit nach links) und 1 (nach rechts) möglich
        self.observation_space = gym.spaces.box.Box(np.array([-100, -100]), np.array([100, 100]),(2,)) # Observationspace beinhaltet die Position des Agenten und die des Ziels
        self.step_counter = 0
        self.cum_reward = 0
        self.destination = 100


    def step(self, action):
        pre_distance = abs(self.destination - self.position)
        self.step_counter += 1
        speed = action[0]
        self.position += speed
        done = False
        distance = abs(self.destination - self.position)


        if distance < pre_distance: reward = 5 * abs(speed)
        else: reward = -5 * abs(speed)

        if distance <= 1:
            reward = 100
            self.position = self.destination
            done = True
            print('Das Ziel wurde erreicht nach {} Schritten. Das Ziel war an Position {}. '.format(self.step_counter, self.destination))
        elif self.position < -100:
            reward = -100
            self.position = -100
            done = True
        elif self.position > 100:
            reward = -100
            self.position = 100
            done = True
        elif self.step_counter >= 200:
            done = True

        self.cum_reward += reward

        if done == True:
            print('Die Episode ist beendet. Kummulierte Belohnung: {}, Anzahl Schritte: {}'.format(self.cum_reward, self.step_counter))

        return ([self.position, self.destination], reward, done, dict())


    def reset(self):
        self.position = 0
        self.cum_reward = 0
        self.step_counter = 0

        if np.random.rand() > 0.5:
            self.destination = 100
        else :
            self.destination = -100


        return [self.position, self.destination]

    def render(self, mode='human'):
        pass



