from pyforce.env import wrap_openai_gym
# from pyforce.nn import default_network_components
from pyforce.agents import TD3Agent
# import gym
import torch
from pathlib import Path
import pickle

import sandbox_env as sandbox_env
# from pyforce import agents

LOAD_AGENT_FROM = None

agent_params = {
    'save_path': "./evals/td3_new",
    'critic_lr': 1e-3,
    'actor_lr': 1e-3,
    'episodes': 6000,
    'train_freq': 1,
    'batch_size': 500,
    'policy_noise': 0.1,
    'policy_noise_clip': .25,
    'gamma': .99,
    'policy_freq': 2,
    'tau': 0.005,
    'warmup_steps': 10000,
    'buffer_size': 50000,
    'exp_noise': .1,
    'eval_freq': 10,
    'render': False,
    'eval_episodes': 10
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

parm_list = [agent_params, sandbox_env.env_params]
file_path = agent_params['save_path'] + '/params'
with open(file_path, 'wb') as f:
    pickle.dump(parm_list, f)
file_path_txt = file_path + '.txt'
with open(file_path_txt, 'w') as f:
    f.write('agent_params\n')
    for k in agent_params.keys():
        f.write('\t' + k + ' = ' + str(agent_params[k]) + '\n')
    f.write('env_params\n')
    for k in sandbox_env.env_params.keys():
        f.write('\t' + k + ' = ' + str(sandbox_env.env_params[k]) + '\n')

device="cuda:0" if torch.cuda.is_available() else "cpu"
# # torch.cuda.set_device(0)
# device= "cpu"
print(device)
env=wrap_openai_gym(sandbox_env.App(always_render=False))
env.to(device)

agent=TD3Agent(
    env,
    save_path=agent_params['save_path'],
    critic_lr=agent_params['critic_lr'],
    actor_lr=agent_params['actor_lr']
).to(device)

# agent=TD3Agent(
#     env,
#     save_path="./evals/td3_example",
#     critic_lr=1e-3,
#     actor_lr=1e-3
# ).to(device)

# agent.train(env,100000,train_freq=1,batch_size=100,policy_noise=0.1,policy_noise_clip=.25,gamma=.99, policy_freq=2, tau=0.005,warmup_steps=10000,buffer_size=50000, exp_noise=.1,eval_freq=1, render=True, eval_episodes=1)
agent.train(env,agent_params['episodes'],train_freq=agent_params['train_freq'],batch_size=agent_params['batch_size'],policy_noise=agent_params['policy_noise'],
            policy_noise_clip=agent_params['policy_noise_clip'], gamma=agent_params['gamma'], policy_freq=agent_params['policy_freq'], tau=agent_params['tau'],
            warmup_steps=agent_params['warmup_steps'],buffer_size=agent_params['buffer_size'], exp_noise=agent_params['exp_noise'],eval_freq=agent_params['eval_freq'],
            render=agent_params['render'], eval_episodes=agent_params['eval_episodes'])