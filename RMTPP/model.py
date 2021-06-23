import torch
import numpy as np
from functools import partial
from scipy.integrate import quad
import json
import torch.nn as nn
import torch.nn.functional as F

class RMTPP(nn.Module):
    def __init__(self, dim=2, embedDim=4, lstm_hidden_dim=256, hidden_dim=128, timeLossRate=0.1): # used to be 32, 16, 0.3
        super(RMTPP, self).__init__()
        
        self.embed = nn.Embedding(dim, embedDim)
        # first do the embed job
        
        self.lstm = nn.LSTM(embedDim + 1, lstm_hidden_dim)
        # lstm is required
        self.hidden = nn.Linear(lstm_hidden_dim, hidden_dim)
        # as normal, pass the hidden layer
        self.ToTime = nn.Linear(hidden_dim, 1)
        self.ToEvent = nn.Linear(hidden_dim, dim)
        # two nets for the prediction
        self.timeLossRate = timeLossRate
        # to combine time loss and event loss
        self.TimeW = nn.Parameter(torch.full((1,), 0.1))
        self.TimeB = nn.Parameter(torch.full((1,), 0.1))
        # finetune the time

    def forward(self, input):
        time, event = input[0], input[1]
        embed = self.embed(event)
        time = time[:, :, None]
        time_event = torch.cat([time, embed], dim=2)
        # init
        hidden, _ = self.lstm(time_event)
        # print(hidden)
        hidden = hidden[:, -1, :]
        hidden = self.hidden(hidden)
        event_output = self.ToEvent(hidden)
        time_output = torch.squeeze(self.ToTime(hidden))
        return time_output, event_output

    def calcLoss(self, time_output, TimeW,TimeB,t):
        return torch.mean(-(time_output + TimeW * t + TimeB + (torch.exp(time_output + TimeB) - torch.exp(time_output + TimeW * t + TimeB)) / TimeW))

    def loss(self, input, target):
        time_output, event_output = self.forward(input)
        time_target, event_target = target
        event_loss = F.cross_entropy(event_output, event_target)

        time_loss = self.calcLoss(time_output, self.TimeW, self.TimeB, time_target)
        return time_loss, event_loss, self.timeLossRate * time_loss + event_loss

    def predict_time(self, TimeW, TimeB, time_output, last_time):
        res = []
        for to, lt in zip(time_output, last_time):
            res.append(lt +
                       quad(lambda t: t * np.exp(to + TimeW * t + TimeB + (np.exp(to + TimeB) - np.exp(to + TimeW * t + TimeB)) / TimeW),
                            a=0.0, b=20.0)[0])
        return torch.tensor(res)

    def generate(self, input):
        time_output, event_output = self.forward(input)
        event_output = event_output.detach()
        time_output = time_output.detach()
        event_choice = torch.argmax(event_output, dim=1)

        last_time = input[2].cpu().numpy()
        time_output = time_output.cpu().numpy()

        TimeW = self.TimeW.detach().cpu().item()
        TimeB = self.TimeB.detach().cpu().item()

        time_predicted = self.predict_time(TimeW, TimeB, time_output, last_time)

        return time_predicted, event_choice

def load_model():
    return RMTPP()