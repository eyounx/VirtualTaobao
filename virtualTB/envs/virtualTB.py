import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
from virtualTB.model.ActionModel import ActionModel
from virtualTB.model.UserModel import UserModel
from virtualTB.utils import *

class VirtualTB(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.n_item = 5
        self.n_user_feature = 88
        self.n_item_feature = 27
        self.max_page_index = 100
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (self.n_user_feature + 3,), dtype = np.float32)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.n_item_feature,), dtype = np.float32)
        self.user_model = UserModel()
        self.user_model.load()
        self.user_action_model = ActionModel()
        self.user_action_model.load()

    def seed(self, sd = 0):
        torch.manual_seed(sd)

    @property
    def state(self):
        return np.concatenate((self.cur_user, self.lst_action), axis = -1)

    def __user_generator(self):
        # with shape(n_user_feature,)
        return self.user_model.generate()

    def step(self, action):
        # Action: tensor with shape (27, )
        self.lst_action = self.user_action_model.predict(FLOAT(self.cur_user).unsqueeze(0), FLOAT(action).unsqueeze(0)).detach().numpy()[0]
        click, pay, page = np.clip(self.lst_action, a_min = 0, a_max = None)
        reward = int(click)
        self.total_click += reward
        self.total_page += page
        self.rend_action = deepcopy(self.lst_action)
        done = (self.total_page > self.max_page_index)
        if done:
            self.cur_user = self.__user_generator().squeeze().detach().numpy()
            self.lst_action = FLOAT([0,0,0])
        return self.state, reward, done, {'total_click':self.total_click}

    def reset(self):
        self.total_click = 0
        self.total_page = 0
        self.cur_user = self.__user_generator().squeeze().detach().numpy()
        self.lst_action = FLOAT([0,0,0])
        self.rend_action = deepcopy(self.lst_action)
        return self.state

    def render(self, mode='human', close=False):
        print('Current State:')
        print('\t', self.state)
        click, pay, page = np.clip(self.rend_action, a_min = 0, a_max = None)
        print('User\'s action:')
        print('\tclick:%2d, leave:%s, page index:%2d' % (int(click), 'True' if page > self.max_page_index else 'False', int(page)))
        print('Total clicks:', self.total_click)