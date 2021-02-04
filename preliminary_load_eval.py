from pyforce.env import wrap_openai_gym
from pyforce.nn import default_network_components
from pyforce.agents import PPOAgent
from pyforce.agents import TD3Agent
import gym
import torch
from pathlib import Path
import pickle
# import preliminary_env as sandbox_env
import sandbox_env
# from pyforce import agents

PPO = True

# LOAD_PATH = './evals/action_space/polar/3/'
#LOAD_PATH = './evals/observation_space/only diff/3/'
#LOAD_PATH = './evals/observation_space_angle/coord+diff/1/'
# LOAD_PATH = './evals/preliminary_minimal/robots_first/4/'
# LOAD_PATH = './evals/reward_function/angle-False/3/'
LOAD_PATH = './evals/final/21/'
# LOAD_PATH = './evals/algorithm/ppo/3/'
file_path = LOAD_PATH + '/params'
with open(file_path, 'rb') as f:
    parm_list = pickle.load(f)

agent_params = parm_list[0]
env_parms = parm_list[1]

# sandbox_env.N_BOTS = env_parms['N_BOTS']

# device="cuda:0" if torch.cuda.is_available() else "cpu"
# # torch.cuda.set_device(0)
device = "cpu"

env=wrap_openai_gym(sandbox_env.App(always_render=True, verbose=False, traj_savepath=LOAD_PATH))

if PPO:

    observation_processor,hidden_layers,action_mapper=default_network_components(env)

    agent=PPOAgent(
        observation_processor,
        hidden_layers,
        action_mapper,
        save_path=agent_params['save_path'],
        value_lr=agent_params['value_lr'],
        policy_lr=agent_params['policy_lr']
    ).to(device)

else:
    agent = TD3Agent(
        env,
        save_path=agent_params['save_path'],
        critic_lr=agent_params['critic_lr'],
        actor_lr=agent_params['actor_lr']
    ).to(device)

file_path = LOAD_PATH + '/agent'
#file_path = LOAD_PATH + '/agent112117.0'
with open(file_path, 'rb') as f:
    state_dict = pickle.load(f)
agent.load_state_dict(state_dict)

if PPO:
    # agent.train(env,episodes=1000,train_freq=2048,eval_freq=50,render=True, batch_size=128,gamma=.99,tau=.95,clip=.2,n_steps=32,entropy_coef=.01)
    agent.eval(env,eval_episodes=40,render=True, episodes=agent_params['episodes'],train_freq=agent_params['train_freq'],eval_freq=agent_params['eval_freq'], batch_size=agent_params['batch_size'],
                gamma=agent_params['gamma'],tau=agent_params['tau'],clip=agent_params['clip'],n_steps=agent_params['n_steps'],entropy_coef=agent_params['entropy_coef'])

else:
    agent.eval(env, eval_episodes=20, episodes=agent_params['episodes'], train_freq=agent_params['train_freq'],
                batch_size=agent_params['batch_size'], policy_noise=agent_params['policy_noise'],
                policy_noise_clip=agent_params['policy_noise_clip'], gamma=agent_params['gamma'],
                policy_freq=agent_params['policy_freq'], tau=agent_params['tau'],
                warmup_steps=agent_params['warmup_steps'], buffer_size=agent_params['buffer_size'],
                exp_noise=agent_params['exp_noise'], eval_freq=agent_params['eval_freq'])