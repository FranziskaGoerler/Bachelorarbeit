from pyforce.env import wrap_openai_gym
from pyforce.nn import default_network_components
from pyforce.agents import PPOAgent
import gym
import torch
from pathlib import Path
import pickle
import shutil
import profile

import preliminary_env
# from pyforce import agents

LOAD_AGENT_FROM = None
# LOAD_AGENT_FROM = './evals/ppo_sandbox/18/agent'

# for space in ['polar', 'cartesian', 'scaled']:
for space in ['coordinates', 'coord+diff', 'only diff']:
    print('')
    print('=====================================')
    print(space)
    print('')
    for i in range(3):
        print(i)

        # preliminary_env.ACTION_SPACE_TYPE = space
        preliminary_env.OBSERVATION_SPACE_TYPE = space

        agent_params = {
            'save_path': "./evals/preliminary/observation_space/{}/".format(space),
            'value_lr': 5e-4,   # original 5e-4
            'policy_lr': 5e-4,  # original 5e-4
            'episodes': 6000,
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

        Path(agent_params['save_path']).mkdir(parents=True, exist_ok=True)

        run_number = 0
        for p in Path(agent_params['save_path']).iterdir():
            if p.is_dir() and p.name.isnumeric():
                if int(p.name) > run_number:
                    run_number = int(str(p.name))
        run_number += 1
        agent_params['save_path'] += ('/' + str(run_number))

        Path(agent_params['save_path']).mkdir(parents=True, exist_ok=True)

        parm_list = [agent_params, preliminary_env.env_params]
        file_path = agent_params['save_path'] + '/params'
        with open(file_path, 'wb') as f:
            pickle.dump(parm_list, f)
        file_path_txt = file_path + '.txt'
        with open(file_path_txt, 'w') as f:
            f.write('agent_params\n')
            for k in agent_params.keys():
                f.write('\t' + k + ' = ' + str(agent_params[k]) + '\n')
            f.write('env_params\n')
            for k in preliminary_env.env_params.keys():
                f.write('\t' + k + ' = ' + str(preliminary_env.env_params[k]) + '\n')

        device = "cpu"

        env=wrap_openai_gym(preliminary_env.App(always_render=False, verbose=False))

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

        agent.train(env,episodes=agent_params['episodes'],train_freq=agent_params['train_freq'],eval_freq=agent_params['eval_freq'],render=agent_params['render'], batch_size=agent_params['batch_size'],
                    gamma=agent_params['gamma'],tau=agent_params['tau'],clip=agent_params['clip'],n_steps=agent_params['n_steps'],entropy_coef=agent_params['entropy_coef'])

