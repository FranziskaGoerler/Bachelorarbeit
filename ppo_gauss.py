from pyforce.env import wrap_openai_gym
from pyforce.nn import default_network_components
from pyforce.agents import PPOAgent
import gym
import torch
from pathlib import Path
import pickle
import shutil
import profile

import gauss_env
# from pyforce import agents

LOAD_AGENT_FROM = None
# LOAD_AGENT_FROM = './evals/ppo_sandbox/18/agent'

agent_params = {
    'save_path': "./evals/ppo_gauss",
    'value_lr': 5e-5,   # original 5e-4
    'policy_lr': 5e-5,  # original 5e-4
    'episodes': 50000,
    'train_freq': 2048,
    'eval_freq': 50,
    'render': False,
    'batch_size': 256,
    'gamma': .99,
    'tau': .95,
    'clip': .2,
    'n_steps': 32,
    'entropy_coef': .01
}

Path(agent_params['save_path']).mkdir(exist_ok=True)

run_number = 0
for p in Path(agent_params['save_path']).iterdir():
    if p.is_dir() and p.name.isnumeric():
        if int(p.name) > run_number:
            run_number = int(str(p.name))
run_number += 1
agent_params['save_path'] += ('/' + str(run_number))

Path(agent_params['save_path']).mkdir(exist_ok=True)

parm_list = [agent_params, gauss_env.env_params]
file_path = agent_params['save_path'] + '/params'
with open(file_path, 'wb') as f:
    pickle.dump(parm_list, f)
file_path_txt = file_path + '.txt'
with open(file_path_txt, 'w') as f:
    f.write('agent_params\n')
    for k in agent_params.keys():
        f.write('\t' + k + ' = ' + str(agent_params[k]) + '\n')
    f.write('env_params\n')
    for k in gauss_env.env_params.keys():
        f.write('\t' + k + ' = ' + str(gauss_env.env_params[k]) + '\n')

# device="cuda:0" if torch.cuda.is_available() else "cpu"
# # torch.cuda.set_device(0)
device = "cpu"

env=wrap_openai_gym(gauss_env.App(always_render=False, verbose=False))

observation_processor,hidden_layers,action_mapper=default_network_components(env)

agent=PPOAgent(
    observation_processor,
    hidden_layers,
    action_mapper,
    save_path=agent_params['save_path'],
    value_lr=agent_params['value_lr'],
    policy_lr=agent_params['policy_lr']
).to(device)

if LOAD_AGENT_FROM is not None:
    with open(LOAD_AGENT_FROM, 'rb') as f:
        state_dict = pickle.load(f)
    agent.load_state_dict(state_dict)

    # agent.eval(env,eval_episodes=20,render=True, episodes=agent_params['episodes'],train_freq=agent_params['train_freq'],eval_freq=agent_params['eval_freq'], batch_size=agent_params['batch_size'],
    #             gamma=agent_params['gamma'],tau=agent_params['tau'],clip=agent_params['clip'],n_steps=agent_params['n_steps'],entropy_coef=agent_params['entropy_coef'])

# agent.train(env,episodes=1000,train_freq=2048,eval_freq=50,render=True, batch_size=128,gamma=.99,tau=.95,clip=.2,n_steps=32,entropy_coef=.01)

agent.train(env,episodes=agent_params['episodes'],train_freq=agent_params['train_freq'],eval_freq=agent_params['eval_freq'],render=agent_params['render'], batch_size=agent_params['batch_size'],
            gamma=agent_params['gamma'],tau=agent_params['tau'],clip=agent_params['clip'],n_steps=agent_params['n_steps'],entropy_coef=agent_params['entropy_coef'])

# command = "agent.train(env,episodes=agent_params['episodes'],train_freq=agent_params['train_freq'],eval_freq=agent_params['eval_freq'],render=agent_params['render'], batch_size=agent_params['batch_size'], gamma=agent_params['gamma'],tau=agent_params['tau'],clip=agent_params['clip'],n_steps=agent_params['n_steps'],entropy_coef=agent_params['entropy_coef'])"
# profile.run(command, filename='gauss_stat.p')

# pycking_env3.IGNORE_ROBOTS = False
#
# path_from = (agent_params['save_path'] + '/agent')
# path_to = (agent_params['save_path'] + '/agent1')
# shutil.copy(path_from, path_to)

# agent.eval(env,eval_episodes=20,render=True, episodes=agent_params['episodes'],train_freq=agent_params['train_freq'],eval_freq=agent_params['eval_freq'], batch_size=agent_params['batch_size'],
#             gamma=agent_params['gamma'],tau=agent_params['tau'],clip=agent_params['clip'],n_steps=agent_params['n_steps'],entropy_coef=agent_params['entropy_coef'])
#
# agent.train(env,episodes=agent_params['episodes'],train_freq=agent_params['train_freq'],eval_freq=agent_params['eval_freq'],render=agent_params['render'], batch_size=agent_params['batch_size'],
#             gamma=agent_params['gamma'],tau=agent_params['tau'],clip=agent_params['clip'],n_steps=agent_params['n_steps'],entropy_coef=agent_params['entropy_coef'])
