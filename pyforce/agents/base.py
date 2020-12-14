import torch
import gym
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ..memory import Memory
from tqdm.auto import tqdm
import pickle


class BaseAgent(nn.Module):
    def __init__(self, save_path='evals/auto'):
        super().__init__()
        self.memory = Memory()
        self.eval_memory = Memory()
        self.env_steps = nn.Parameter(torch.zeros(1)[0], requires_grad=False)
        self.env_episodes = nn.Parameter(torch.zeros(1)[0], requires_grad=False)
        self.eval_env_steps = nn.Parameter(torch.zeros(1)[0], requires_grad=False)
        self.eval_env_episodes = nn.Parameter(torch.zeros(1)[0], requires_grad=False)
        self.writer = None
        self.save_path = save_path
        self.write_tensorboard_files = False     # is set to True when train is called, so tensorboard file is not written when only evaluation is done

        # if save_path is not None:
        #     self.load(save_path)

    def load(self):
        self.writer = SummaryWriter(self.save_path, flush_secs=10)
        pass

    def write_scalar(self, tag, value, step=None):
        if self.write_tensorboard_files:
            if self.writer is None:
                self.load()
            step = self.env_steps if step is None else step
            self.writer.add_scalar(tag, value, step)

    def write_scalars(self, main_tag, tag_scalar_dict, step=None):
        if self.write_tensorboard_files:
            if self.writer is None:
                self.load()
            step = self.env_steps if step is None else step
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def store_agent(self, store_additional=False):
        if self.save_path is not None:
            save_file = self.save_path + '/agent'
            with open(save_file, 'wb') as f:
                pickle.dump(self.state_dict(), f)
            if store_additional:
                save_file2 = self.save_path + '/agent{}'.format(self.env_steps)
                with open(save_file2, 'wb') as f:
                    pickle.dump(self.state_dict(), f)

    def load_agent_params(self):
        if self.save_path is not None:
            save_file = self.save_path + '/agent'
            with open(save_file, 'rb') as f:
                state_dict = pickle.load(f)
            self.load_state_dict(state_dict)

    def train(self, env, episodes=1000, eval_freq=None, eval_env=None, **kwargs):
        self.write_tensorboard_files = True
        store_additional = False
        for episode in tqdm(range(episodes)):
            done = False
            state = env.reset()
            while not done:
                action, action_info = self.get_action(state, False, kwargs)
                next_state, reward, done, _ = env.step(action)
                self.memory.append(state=state, action=action, next_state=next_state, reward=reward, done=done, **action_info)
                state = next_state
                done = done.max() == 1
                self.env_steps += 1
                if self.env_steps % 100000 == 0:
                    store_additional = True
                self.after_step(done, False, kwargs)

            self.env_episodes += 1

            if eval_freq is not None and self.env_episodes % eval_freq == 0 and ("warmup_steps" not in kwargs or len(self.memory) > kwargs["warmup_steps"]):
                self.store_agent(store_additional=store_additional)
                store_additional = False
                self.eval(env if eval_env is None else eval_env, episodes=1, **kwargs)

        self.store_agent()   # store at the end, so it is stored also when eval is off

    def eval(self, env, eval_episodes=10, render=False, **kwargs):
        for episode in range(eval_episodes):
            done = False
            state = env.reset()
            while not done:
                if render:
                    env.render()
                action, action_info = self.get_action(state, True, kwargs)
                next_state, reward, done, _ = env.step(action)
                self.eval_memory.append(state=state, action=action, next_state=next_state, reward=reward, done=done, **action_info)
                state = next_state
                done = done.max() == 1
                self.eval_env_steps += 1
                self.after_step(done, True, kwargs)

            self.eval_env_episodes += 1

    def get_action(self, state, eval, args):
        raise NotImplementedError()

    def after_step(self, done, eval, args):
        raise NotImplementedError()
