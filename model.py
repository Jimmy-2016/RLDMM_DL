
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, emd_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(1, emd_dim)
        self.fc2 = nn.Linear(emd_dim, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


def TakeAction(action, st, ev, Reward, Dtemp):
    if Dtemp == -1:
        Dtemp = 0

    if action == 2:
        # NexState = torch.round(st + ev)
        NexState = st + ev

    else:
        NexState = st

    # if NexState >= 200:
    #     NexState = 200
    # if NexState < 0:
    #     NexState = 0

    FedBack = 2
    done = False
    if action == 1 or action == 0:
        done = True
        if action == Dtemp:
            FedBack = 0
        else:
            FedBack = 1

    return torch.tensor(NexState, dtype=torch.float), torch.tensor(Reward[FedBack]), torch.tensor(done)