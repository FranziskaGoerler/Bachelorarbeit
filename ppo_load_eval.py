from pyforce.env import wrap_openai_gym
from pyforce.nn import default_network_components
from pyforce.agents import PPOAgent
import gym
import torch
from pathlib import Path
import pickle

import pycking_env3
# from pyforce import agents

LOAD_PATH = './evals/ppo_pycking/5/'

file_path = LOAD_PATH + '/params'
with open(file_path, 'rb') as f:
    parm_list = pickle.load(f)

agent_params = parm_list[0]
env_parms = parm_list[1]

pycking_env3.N_BOTS = env_parms['N_BOTS']

# device="cuda:0" if torch.cuda.is_available() else "cpu"
# # torch.cuda.set_device(0)
device = "cpu"

env=wrap_openai_gym(pycking_env3.App(always_render=True, verbose=False))

observation_processor,hidden_layers,action_mapper=default_network_components(env)

agent=PPOAgent(
    observation_processor,
    hidden_layers,
    action_mapper,
    save_path=agent_params['save_path'],
    value_lr=agent_params['value_lr'],
    policy_lr=agent_params['policy_lr']
).to(device)

file_path = LOAD_PATH + '/agent'
with open(file_path, 'rb') as f:
    state_dict = pickle.load(f)
agent.load_state_dict(state_dict)

# agent.train(env,episodes=1000,train_freq=2048,eval_freq=50,render=True, batch_size=128,gamma=.99,tau=.95,clip=.2,n_steps=32,entropy_coef=.01)
agent.eval(env,eval_episodes=1000,render=True, episodes=agent_params['episodes'],train_freq=agent_params['train_freq'],eval_freq=agent_params['eval_freq'], batch_size=agent_params['batch_size'],
            gamma=agent_params['gamma'],tau=agent_params['tau'],clip=agent_params['clip'],n_steps=agent_params['n_steps'],entropy_coef=agent_params['entropy_coef'])