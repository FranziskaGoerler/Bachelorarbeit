import gym
import numpy as np

class App(gym.Env):
    def __init__(self):
        self.position = 0
        self.action_space = gym.spaces.box.Box(np.array([-1]), np.array([1]), (1,)) # Aktionen zwischen -1 (mit max. Geschwindigkeit nach links) und 1 (nach rechts) möglich
        self.observation_space = gym.spaces.box.Box(np.array([-100]), np.array([100]),(1,)) # Observationspace beinhaltet die Position des Agenten in der Umgebung von -100 bis 100
        self.step_counter = 0
        self.cum_reward = 0


    def step(self, action):
        self.step_counter += 1
        speed = action[0]
        self.position += speed
        done = False


        if speed > 0: reward = 5 * speed
        else: reward = 0

        if self.position >= 100:
            reward = 100
            self.position = 100
            done = True
            print('Das Ziel wurde erreicht nach {} Schritten '.format(self.step_counter))
        elif self.position < -100:
            reward = -100
            self.position = -100
            done = True
        elif self.step_counter >= 200:
            done = True

        self.cum_reward += reward

        if done == True:
            print('Die Episode ist beendet. Kummulierte Belohnung: {}, Anzahl Schritte: {}'.format(self.cum_reward, self.step_counter))

        return (self.position, reward, done, dict())


    def reset(self):
        self.position = 0
        self.cum_reward = 0
        self.step_counter = 0
        return [self.position]

    def render(self, mode='human'):
        pass
