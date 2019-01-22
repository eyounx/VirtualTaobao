import os
from virtualTB.utils import *

class ActionModel(nn.Module):
    def __init__(self, n_input = 115, n_output = 30 + 10 + 500, learning_rate = 0.01):
        super(ActionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)
        )   
        self.max_click = 30
        self.max_deal = 10

    def predict(self, user, weight):
        x = self.model(torch.cat((user, weight), dim = -1))
        click = torch.multinomial(F.softmax(x[:, :self.max_click], dim = 1), 1)
        pay = torch.multinomial(F.softmax(x[:, self.max_click:self.max_click + self.max_deal], dim = 1), 1)
        page = torch.multinomial(F.softmax(x[:, self.max_click + self.max_deal:], dim = 1), 1) + 1
        return torch.cat((click, pay, page), dim = -1)

    def save(self, path = None):
        if path == None:
            path = os.path.dirname(__file__) + '/../data/action_model.pt'
        torch.save(self.model.state_dict(), path)

    def load(self, path = None):
        if path == None:
            path = os.path.dirname(__file__) + '/../data/action_model.pt'
        self.model.load_state_dict(torch.load(path))
