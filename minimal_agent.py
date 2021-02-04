from pyforce.env import wrap_openai_gym
# from pyforce.nn import default_network_components
from pyforce.agents import TD3Agent
# import gym
import torch
from pathlib import Path

import minimal_v4 as envi
# import minimal_v1
# from pyforce import agents

save_path = "./evals/td3_example"
description = 'Minimal 4: Umgebung im Bereich von (-100,-100) bis (100,100), Observationspace: Eigene Koordinaten und Zielposition, Startposition(0,0), Zufällige Zielposition'
# description = 'minimal_v4'
device="cuda:0" if torch.cuda.is_available() else "cpu"
# # torch.cuda.set_device(0)
# device= "cpu"
# print(device)
# env=wrap_openai_gym(minimal_v1.App())
env=wrap_openai_gym(envi.App())

Path(save_path).mkdir(exist_ok=True)

run_number = 0
for p in Path(save_path).iterdir():
    if p.is_dir() and p.name.isnumeric():
        if int(p.name) > run_number:
            run_number = int(str(p.name))
run_number += 1
save_path += ('/' + str(run_number))

Path(save_path).mkdir(exist_ok=True)

with open(save_path+'/description.txt', mode='w') as f:
    f.write(description)

agent=TD3Agent(
    env,
    save_path=save_path,
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
agent.train(env,500,train_freq=1,batch_size=800,policy_noise=0.1,policy_noise_clip=.25,gamma=.99, policy_freq=2, tau=0.005,warmup_steps=10000,buffer_size=50000, exp_noise=.1,eval_freq=10, render=True, eval_episodes=10)