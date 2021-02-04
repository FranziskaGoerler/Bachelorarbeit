from pyforce.env import wrap_openai_gym
# from pyforce.nn import default_network_components
from pyforce.agents import TD3Agent
# import gym
import torch
from pathlib import Path
import pickle

import minimal_v9 as envi
# import minimal_v1
# from pyforce import agents

LOAD_PATH = './evals/td3_example/14/'

device="cuda:0" if torch.cuda.is_available() else "cpu"
# # torch.cuda.set_device(0)
# device= "cpu"
# print(device)
# env=wrap_openai_gym(minimal_v1.App())
env=wrap_openai_gym(envi.App())

agent=TD3Agent(
    env,
    save_path='',
    critic_lr=1e-3,
    actor_lr=1e-3
).to(device)

file_path = LOAD_PATH + '/agent'
with open(file_path, 'rb') as f:
    state_dict = pickle.load(f)
agent.load_state_dict(state_dict)

# agent=TD3Agent(
#     env,
#     save_path="./evals/td3_example",
#     critic_lr=1e-3,
#     actor_lr=1e-3
# ).to(device)

# agent.train(env,100000,train_freq=1,batch_size=100,policy_noise=0.1,policy_noise_clip=.25,gamma=.99, policy_freq=2, tau=0.005,warmup_steps=10000,buffer_size=50000, exp_noise=.1,eval_freq=1, render=True, eval_episodes=1)
agent.eval(env,eval_episodes=1000,train_freq=1,batch_size=800,policy_noise=0.1,policy_noise_clip=.25,gamma=.99, policy_freq=2, tau=0.005,warmup_steps=10000,buffer_size=50000, exp_noise=.1,eval_freq=10, render=True)