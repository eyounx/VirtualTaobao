import os
import io
import time
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import virtualTB
import numpy as np

FLOAT = torch.FloatTensor

def init_weight(m):
    if type(m) == nn.Linear:
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(88 + 3, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 27),
            nn.Tanh()
        )
        self.model.apply(init_weight)
    
    def forward(self, x):
        return self.model(x)

def train(state, action, label, batch_size = 100):
    N = len(label)
    loss_func = nn.MSELoss()
    batch_num = (len(label) + batch_size - 1) // batch_size
    for epoch in range(1000):
        idx = np.random.permutation(N)
        total_loss = 0
        for i in range(batch_num):
            batch_idx = idx[i*batch_size:(i+1)*batch_size]
            m_state, m_action, m_label = state[batch_idx], action[batch_idx], label[batch_idx]
            y_action = model(m_state)
            loss = torch.mean(m_label * ((y_action - m_action) ** 2).sum(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().numpy() / batch_num
        ctr = test()
        print('Epoch %3d: Loss %.2f CTR: %.2f' % (epoch, total_loss, ctr))

def test():
    total_reward = 0
    total_page = 0
    for i in range(50):
        state = env.reset()
        done = False
        while not done:
            action = model(FLOAT(state))
            # action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_page += 1
            if done:
                break
    ctr = total_reward / total_page / 10
    return ctr

if __name__ == '__main__':
    env = gym.make('VirtualTB-v0')
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    state, action, label = [], [], []
    with io.open(os.path.dirname(__file__) + '/../data/dataset.txt','r') as file:
        for line in file:
            state_l, action_l, label_l = line.split('\t')
            state.append([float(x) for x in state_l.split(',')])
            action.append([float(x) for x in action_l.split(',')])
            label.append(int(label_l))
    state, action, label = FLOAT(state), FLOAT(action), FLOAT(label)
    train(state, action, label)
    CTR = test()
    print('CTR: %.2f' % CTR)

