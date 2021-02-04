from pyforce.env import wrap_openai_gym
from pyforce.nn import default_network_components
from pyforce.agents import PPOAgent
from pyforce.agents import TD3Agent
import gym
import torch
from pathlib import Path
import pickle
import shutil
import profile
import itertools
import preliminary_env
# import minimal_preliminary

# from pyforce import agents

LOAD_AGENT_FROM = None
# LOAD_AGENT_FROM = './evals/ppo_sandbox/18/agent'

# device="cuda:0" if torch.cuda.is_available() else "cpu"
# # torch.cuda.set_device(0)
device = "cpu"
print(device)

# ag = 'td3'
# for space in ['scaled', 'cartesian', 'polar']:
# for space in ['coordinates', 'coord+diff', 'only diff']:
# for space in ['coordinates', 'coord+diff']:
# for fun, pun in itertools.product(['angle', 'distance'], [False, True]):
# for ag in ['ppo', 'td3']:

for ag in ['ppo']:
    print('')
    print('=====================================')
    print(ag)
    print('')
    for i in range(1):
        print(i)

        # preliminary_env.ACTION_SPACE_TYPE = space
        # preliminary_env.OBSERVATION_SPACE_TYPE = space
        # minimal_preliminary.REWARD_FUNCTION = fun
        # minimal_preliminary.PUNISH_WRONG_DIRECTION = pun

        if ag == 'ppo':
            agent_params = {
                'save_path': "./evals/preliminary/algorithm_angle/{}/".format(ag),
                'value_lr': 5e-4,
                'policy_lr': 5e-4,
                'episodes': 35000,
                'train_freq': 2048,
                'eval_freq': 50,
                'render': False,
                'batch_size': 256,
                'gamma': .99,
                'tau': .95,
                'clip': .2,
                'n_steps': 32,
                'entropy_coef': .01,
                'store_agent_every': 100000
            }
        else:
            agent_params = {
                'save_path': "./evals/preliminary/algorithm_angle/{}/".format(ag),
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
                'eval_episodes': 10,
                'store_agent_every': 10000
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
        # env=wrap_openai_gym(minimal_preliminary.App())

        if ag == 'ppo':
            observation_processor, hidden_layers, action_mapper = default_network_components(env)
            agent=PPOAgent(
                observation_processor,
                hidden_layers,
                action_mapper,
                save_path=agent_params['save_path'],
                value_lr=agent_params['value_lr'],
                policy_lr=agent_params['policy_lr']
            ).to(device)

            agent.train(env, episodes=agent_params['episodes'], train_freq=agent_params['train_freq'],
                        eval_freq=agent_params['eval_freq'], render=agent_params['render'],
                        batch_size=agent_params['batch_size'],
                        gamma=agent_params['gamma'], tau=agent_params['tau'], clip=agent_params['clip'],
                        n_steps=agent_params['n_steps'], entropy_coef=agent_params['entropy_coef'],
                        store_agent_every=agent_params['store_agent_every'])
        else:
            agent = TD3Agent(
                env,
                save_path=agent_params['save_path'],
                critic_lr=agent_params['critic_lr'],
                actor_lr=agent_params['actor_lr']
            ).to(device)

            agent.train(env, agent_params['episodes'], train_freq=agent_params['train_freq'],
                        batch_size=agent_params['batch_size'], policy_noise=agent_params['policy_noise'],
                        policy_noise_clip=agent_params['policy_noise_clip'], gamma=agent_params['gamma'],
                        policy_freq=agent_params['policy_freq'], tau=agent_params['tau'],
                        warmup_steps=agent_params['warmup_steps'], buffer_size=agent_params['buffer_size'],
                        exp_noise=agent_params['exp_noise'], eval_freq=agent_params['eval_freq'],
                        render=agent_params['render'], eval_episodes=agent_params['eval_episodes'],
                        store_agent_every=agent_params['store_agent_every'])

        # if LOAD_AGENT_FROM is not None:
        #     with open(LOAD_AGENT_FROM, 'rb') as f:
        #         state_dict = pickle.load(f)
        #     agent.load_state_dict(state_dict)

