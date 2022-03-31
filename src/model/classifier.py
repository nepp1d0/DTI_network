import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size, **config):
        super(Classifier, self).__init__()
        if 'hidden_layers' in config.keys():
            self.hidden_layers = config['hidden_layers']
        else:
            self.hidden_layers = [512, 64, 32, 1]
        self.fc1 = nn.Linear(input_size, self.hidden_layers[0])
        self.predictor = nn.ModuleList([nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]) for i in range(len(self.hidden_layers)-1)])

    def forward(self, x):
        x = self.fc1(x)
        for i, l in enumerate(self.predictor):
            x = l(x)
        return x
