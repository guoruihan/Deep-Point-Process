import torch
import numpy as np
from functools import partial
from scipy.integrate import quad
import json
import torch.nn as nn
import torch.nn.functional as F



class RMTPP(nn.Module):
    def __init__(self, dim=2, embedDim=4, lstm_hidden_dim=256, hidden_dim=128, timeLossRate=0.3): # used to be 32, 16, 0.3
        super(RMTPP, self).__init__()
        self.embed = nn.Embedding(dim, embedDim)
        self.embed_dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedDim + 1, lstm_hidden_dim)
        self.hidden = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.Linear_T = nn.Linear(hidden_dim, 1)
        self.Linear_E = nn.Linear(hidden_dim, dim)
        self.timeLossRate = timeLossRate
        self.w = nn.Parameter(torch.full((1,), 0.1))
        self.b = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, input):
        time = input[0]
        event = input[1]
        embed = self.embed(event)
        time = time[:, :, None]
        time_event = torch.cat([time, embed], dim=2)
        hidden, _ = self.lstm(time_event)  # seq_len x batch x hidden_dim
        hidden = hidden[:, -1, :]
        hidden = self.hidden(hidden)
        event_output = self.Linear_E(hidden)
        time_output = torch.squeeze(self.Linear_T(hidden))
        return time_output, event_output

    def loss(self, input, target):
        time_output, event_output = self.forward(input)
        time_target, event_target = target
        event_loss = F.cross_entropy(event_output, event_target)

        def time_nll(to, w, b, t):
            return -(to + w * t + b + (torch.exp(to + b) - torch.exp(to + w * t + b)) / w)

        time_loss = torch.mean(time_nll(time_output, self.w, self.b, time_target))
        merged_loss = self.timeLossRate * time_loss + event_loss
        return time_loss, event_loss, merged_loss

    def predict_time(self, w, b, time_output, last_time):
        res = []
        for to, lt in zip(time_output, last_time):
            res.append(lt +
                       quad(lambda t: t * np.exp(to + w * t + b + (np.exp(to + b) - np.exp(to + w * t + b)) / w),
                            a=0.0, b=20.0)[0])
        return torch.tensor(res)

    def generate(self, input):
        time_output, event_output = self.forward(input)
        event_output = event_output.detach()
        time_output = time_output.detach()
        event_choice = torch.argmax(event_output, dim=1)

        last_time = input[2].cpu().numpy()
        time_output = time_output.cpu().numpy()

        w = self.w.detach().cpu().item()
        b = self.b.detach().cpu().item()

        time_predicted = self.predict_time(w, b, time_output, last_time)

        return time_predicted, event_choice

def load_model(name, args):
    name = name.lower()
    if name == 'rmtpp':
        return RMTPP()
    else:
        raise ValueError()