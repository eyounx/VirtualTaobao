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

def train(features, labels, clicks, batch_size = 100):
    N = len(clicks) # number of instances
    loss_func = nn.MSELoss()
    batch_num = (len(clicks) + batch_size - 1) // batch_size
    for epoch in range(1000):
        idx = np.random.permutation(N)
        total_loss = 0
        for i in range(batch_num):
            batch_idx = idx[i*batch_size:(i+1)*batch_size]
            m_features, m_labels, m_clicks = features[batch_idx], labels[batch_idx], clicks[batch_idx]
            y_labels = model(m_features) # predict the labels from the features
            # loss function. note that the loss naturally does not count the instances with zero clicks
            loss = torch.mean(m_clicks * ((y_labels - m_labels) ** 2).sum(1)) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().numpy() / batch_num
        ctr = test()
        print('Epoch %3d: Loss %.2f CTR: %.2f' % (epoch, total_loss, ctr))

def test():
	  # test the model in the interactive system
    total_clicks = 0
    total_page = 0
    for i in range(50):
        features = env.reset()
        done = False
        while not done:
            predictions = model(FLOAT(features))
            features, clicks, done, info = env.step(predictions)
            total_clicks += clicks
            total_page += 1
            if done:
                break
    ctr = total_clicks / total_page / 10
    return ctr

if __name__ == '__main__':
    env = gym.make('VirtualTB-v0')
    # prepare a neural network model
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    # load dataset
    features, labels, clicks = [], [], []
    with io.open(os.path.dirname(__file__) + '/dataset.txt','r') as file:
        for line in file:
            features_l, labels_l, clicks_l = line.split('\t')
            features.append([float(x) for x in features_l.split(',')])
            labels.append([float(x) for x in labels_l.split(',')])
            clicks.append(int(clicks_l))
    # train model
    features, labels, clicks = FLOAT(features), FLOAT(labels), FLOAT(clicks)
    train(features, labels, clicks)
    # test model
    CTR = test()
    print('CTR: %.2f' % CTR)

