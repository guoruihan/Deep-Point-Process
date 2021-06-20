import torch
import numpy as np
from functools import partial
from scipy.integrate import quad
import json
import torch.nn as nn
import torch.nn.functional as F



class RMTPP(nn.Module):
    def __init__(self, event_classes=2, event_embed_dim=4, lstm_hidden_dim=256, hidden_dim=128, loss_alpha=0.3): # used to be 32, 16, 0.3
        super(RMTPP, self).__init__()
        self.event_embedding = nn.Embedding(event_classes, event_embed_dim)
        self.event_embedding_dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(event_embed_dim + 1, lstm_hidden_dim)
        self.hidden_linear = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.time_linear = nn.Linear(hidden_dim, 1)
        self.event_linear = nn.Linear(hidden_dim, event_classes)
        self.loss_alpha = loss_alpha
        self.w = nn.Parameter(torch.full((1,), 0.1))
        self.b = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, input):
        time = input[0]
        event = input[1]
        event_embedding = self.event_embedding(event)
        time = time[:, :, None]
        time_event = torch.cat([time, event_embedding], dim=2)
        hidden, _ = self.lstm(time_event)  # seq_len x batch x hidden_dim
        hidden = hidden[:, -1, :]
        hidden = self.hidden_linear(hidden)
        event_output = self.event_linear(hidden)
        time_output = torch.squeeze(self.time_linear(hidden))
        return time_output, event_output

    def loss(self, input, target):
        time_output, event_output = self.forward(input)
        time_target, event_target = target
        event_loss = F.cross_entropy(event_output, event_target)

        def time_nll(to, w, b, t):
            return -(to + w * t + b + (torch.exp(to + b) - torch.exp(to + w * t + b)) / w)

        time_loss = torch.mean(time_nll(time_output, self.w, self.b, time_target))
        merged_loss = self.loss_alpha * time_loss + event_loss
        return time_loss, event_loss, merged_loss

    def inference(self, input):
        time_output, event_output = self.forward(input)
        event_output = event_output.detach()
        time_output = time_output.detach()
        event_choice = torch.argmax(event_output, dim=1)

        last_time = input[2].cpu().numpy()
        time_output = time_output.cpu().numpy()

        w = self.w.detach().cpu().item()
        b = self.b.detach().cpu().item()

        time_predicted = torch.tensor([ # quad here is for integral
            lt + quad(lambda t: t * np.exp(to + w * t + b + (np.exp(to + b) - np.exp(to + w * t + b)) / w), a=0.0, b=10.0)[0]
            for to, lt
            in zip(time_output, last_time)
        ])

        return time_predicted, event_choice

def load_model(name, args):
    name = name.lower()
    if name == 'rmtpp':
        return RMTPP()
    else:
        raise ValueError()