
import torch
import torch.optim as optim
import numpy as np
from model import *
import random
import matplotlib.pyplot as plt
## Params
num_epochs = 100
emd_dim = 20
##
epsilon = 0.0
lr = 0.001  # learning rate
gamma = 1  # discount factor
K = .04
SigmaEv = 1
States = np.arange(201)
Numstates = len(States)
Actions = [-1, 1, 0]
TerminatingActions = [-1, 1]
Reward = [100, -1000, -1]  # 100, -1000, -1
NumActions = len(Actions)
NumTr = 300
Coherence = np.array([0, 3.2, 6.4, 12.8, 25.6, 51.2])/100
DiVector = np.hstack((np.ones(int(NumTr/2)), -1*np.ones(int(NumTr/2))))
random.shuffle(DiVector)
AllCoh = np.repeat(Coherence, NumTr/len(Coherence))
random.shuffle(AllCoh)
##

policy = PolicyNet(torch.tensor(emd_dim), torch.tensor(NumActions))
optimizer = optim.Adam(policy.parameters(), lr=lr)

all_loss = []
reset = 1
trCount = 0
epochs_rt = []
all_rt = 0
for iter in range(num_epochs):
    trCount = 0
    reset = 1
    st = int(np.floor(len(States) / 2))
    epoch_loss = []
    epochs_rt.append(np.mean(all_rt))
    all_rt = np.zeros(len(AllCoh))

    while True:
        saved_log_probs = []
        rewards = []
        if reset:
            Ctemp = AllCoh[trCount]
            Dtemp = DiVector[trCount]
            # st = torch.tensor(int(np.floor(len(States) / 2)), dtype=torch.float).unsqueeze(0)
            st = torch.tensor(0.0).unsqueeze(0)
            # st = torch.tensor(torch.randint(low=50, high=150, size=(1,))[0], dtype=torch.int32)
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            saved_log_probs = []
            tmprt = 0

        done = False
        reset = 0

        while not done:
            action_probs = policy(st)
            # distribution = torch.distributions.Categorical(action_probs)
            # action = distribution.sample().view(-1)
            action = action_probs.multinomial(1).cpu().numpy()[0]
            saved_log_probs.append(torch.log(action_probs.squeeze(0)[action]))
            # saved_log_probs.append(torch.log(action_probs[action]))
            ev = np.random.normal(K * Dtemp * Ctemp, SigmaEv)
            state, reward, done = TakeAction(action.item(), st, ev, Reward, Dtemp)
            rewards.append(reward)
            st = state
            tmprt += 1

        # Compute returns
        returns = []
        G = 0
        for r in rewards[::-1]:
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float)
        if returns.shape[0] > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Policy gradient update
        policy_loss = []
        for log_prob, G in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * G)

        try:
            loss = torch.stack(policy_loss).mean()
        except:
            letmeknow = 1
        if loss < 0:
            letmeknow = 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_rt[trCount] = tmprt
        trCount += 1
        reset = 1
        if trCount >= len(AllCoh):
            break

    all_loss.append(loss.item())

    if iter % 10 == 0:
        print(f'iter {iter}, Loss: {loss.item()}')


    # print(np.mean(policy_loss))


##
plt.figure()
plt.plot(all_loss)

plt.figure()
xlim = torch.linspace(-1, 1, 100)
action_model_probs = np.zeros((100, 3))
for i in range(xlim.shape[0]):
    action_model_probs[i, :] = policy(xlim[i].unsqueeze(0)).detach()

plt.plot(xlim.detach(), action_model_probs)
plt.xlabel('states')
plt.ylabel('probability')



plt.figure()
plt.plot(all_rt)

plt.figure()
plt.plot(epochs_rt[1:])
plt.xlabel('trials')
plt.ylabel('RT')

plt.show()


