import os
import numpy as np
from virtualTB.utils import *

class UserModel(nn.Module):
    def __init__(self, instance_dimesion=88, seed_dimesion = 128, n_hidden = 128, learning_rate=0.001):
        super(UserModel, self).__init__()
        self.seed_dimesion = seed_dimesion
        self.generator_model = nn.Sequential(
            nn.Linear(seed_dimesion, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, instance_dimesion),
        )
        self.generator_model.apply(init_weight)

    def generator(self, z):
        x = self.generator_model(z)
        return self.softmax_feature(x)
    
    def softmax_feature(self, x):
        features = [None] * 11
        features[0] = x[:, 0:8]
        features[1] = x[:, 8:16]
        features[2] = x[:, 16:27]
        features[3] = x[:, 27:38]
        features[4] = x[:, 38:49]
        features[5] = x[:, 49:60]
        features[6] = x[:, 60:62]
        features[7] = x[:, 62:64]
        features[8] = x[:, 64:67]
        features[9] = x[:, 67:85]
        features[10] = x[:, 85:88]
        entropy = 0
        softmax_feature = FLOAT([])
        for i in range(11):
            softmax_feature = torch.cat((softmax_feature, F.softmax(features[i], dim = 1)), dim = -1)
            entropy += -(F.log_softmax(features[i], dim = 1) * F.softmax(features[i], dim = 1)).sum(dim = 1).mean()
        return softmax_feature, entropy

    def generate(self, z = None):
        if z == None:
            z = torch.rand((1, self.seed_dimesion))
        x = self.generator(z)[0]
        features = [None] * 11
        features[0] = x[:, 0:8] 
        features[1] = x[:, 8:16] 
        features[2] = x[:, 16:27]
        features[3] = x[:, 27:38] 
        features[4] = x[:, 38:49]
        features[5] = x[:, 49:60]
        features[6] = x[:, 60:62]
        features[7] = x[:, 62:64]
        features[8] = x[:, 64:67]
        features[9] = x[:, 67:85]
        features[10] = x[:, 85:88]
        one_hot = FLOAT()
        for i in range(11):
            tmp = torch.zeros(features[i].shape)
            one_hot = torch.cat((one_hot, tmp.scatter_(1, torch.multinomial(features[i], 1), 1)), dim = -1)
        return one_hot

    def load(self, path = None):
        if path == None:
            g_path = os.path.dirname(__file__) + '/../data/generator_model.pt'
        self.generator_model.load_state_dict(torch.load(g_path))
