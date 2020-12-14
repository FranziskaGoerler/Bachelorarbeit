from pyforce.env import wrap_openai_gym
# from pyforce.nn import default_network_components
from pyforce.agents import TD3Agent
# import gym
import torch

import pycking_env3
# from pyforce import agents

# device="cuda:0" if torch.cuda.is_available() else "cpu"
# # torch.cuda.set_device(0)
device= "cpu"
# print(device)
env=wrap_openai_gym(pycking_env3.App(always_render=True))

agent=TD3Agent(
    env,
    save_path="./evals/traj_test2",
    critic_lr=1e-3,
    actor_lr=1e-3
).to(device)

# agent=TD3Agent(
#     env,
#     save_path="./evals/td3_example",
#     critic_lr=1e-3,
#     actor_lr=1e-3
# ).to(device)

# agent.train(env,100000,train_freq=1,batch_size=100,policy_noise=0.1,policy_noise_clip=.25,gamma=.99, policy_freq=2, tau=0.005,warmup_steps=10000,buffer_size=50000, exp_noise=.1,eval_freq=1, render=True, eval_episodes=1)
agent.train(env,100000,train_freq=1,batch_size=800,policy_noise=0.1,policy_noise_clip=.25,gamma=.99, policy_freq=2, tau=0.005,warmup_steps=10000,buffer_size=50000, exp_noise=.1,eval_freq=10, render=True, eval_episodes=1)