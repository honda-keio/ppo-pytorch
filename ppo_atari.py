import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
import torch.multiprocessing as mp
import numpy as np
import gym
import copy, random, time
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from ActorCritic import ActorCriticCNN
from wrappers import make_env

class PPO:
    def __init__(self, ENV, max_epochs, N, T, batch_size, optimizer=optim.Adam, n_mid=10, 
                epsilon=0.2, gamma=0.99, v_loss_coef=0.5, max_grad_norm=0.5, lr=0.01, gpu=None):
        self.envs = SubprocVecEnv([make_env(ENV) for _ in range(N)])
        self.ob_s = self.envs.observation_space.shape
        ac_s = self.envs.action_space.n
        self.model = ActorCriticCNN(self.ob_s, ac_s)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs
        self.N = N
        self.T = T
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.batch_num = self.T * self.N // self.batch_size
        self.v_loss_coef = v_loss_coef
        self.max_grad_norm = max_grad_norm
        if gpu:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def reset_seed(self, seed=1):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    def non_multi(self):
        self.states = torch.zeros([self.T+1, self.N, *self.ob_s]).to(self.device)
        masks = np.ones([self.T+1, self.N, 1], dtype=np.float32)
        rewards = np.zeros([self.T, self.N, 1], dtype=np.float32)
        actions = np.zeros([self.T, self.N, 1], dtype=np.int32)
        states_np = self.envs.reset()
        self.states[0] = torch.from_numpy(states_np)
        masks[0] *= 0
        with torch.no_grad():
            for t in range(self.T):
                actions[t] = self.model.act(self.states[t]).to("cpu").numpy()
                states_np, rewards[t,:,0], dones, _ = self.envs.step(actions[t])
                if dones.sum() != 0:
                    for n in range(self.N):
                        if dones[n]:
                            self.envs.remotes[n].send(("reset", None))
                            states_np[n] = self.envs.remotes[n].recv()
                            masks[t+1,n] *= 0
                self.states[t+1] = torch.from_numpy(states_np)
            last_v = self.model.value(self.states[-1]).to("cpu").numpy()
            returns = np.zeros([self.T, self.N, 1], dtype=np.float32)
            returns[-1] = rewards[-1] + self.gamma * masks[-1] * last_v
            for t in reversed(range(self.T-1)):
                returns[t] = rewards[t] + self.gamma * masks[t+1] * returns[t+1]
        self.posi = False if (rewards < 0).sum() else True
        self.states = self.states[:-1].view(-1, *self.ob_s)
        self.actions = torch.from_numpy(actions).long().view(-1, 1)
        self.returns = torch.from_numpy(returns).view(-1, 1)
        self.reward_mean = rewards.sum(0).mean()

    def update(self):
        self.non_multi()
        old_model = copy.deepcopy(self.model)
        rand_list = torch.randperm(self.batch_num*self.batch_size).view(-1, self.batch_size).tolist()
        for ind in rand_list:
            batch = self.states[ind]
            v, pi = self.model(batch)
            pi_log_prob = F.log_softmax(pi, dim=1)
            with torch.no_grad():
                _, pi_old = old_model(batch)
                pi_old_log_prob = F.log_softmax(pi_old, dim=1)
            A = self.returns[ind].to(self.device) - v
            action = self.actions[ind].to(self.device)
            pi_old_log_prob = pi_old_log_prob.gather(1, action)
            pi_log_prob = pi_log_prob.gather(1, action)
            r = (pi_log_prob - pi_old_log_prob).clamp(max=3).exp()
            clip = r.clamp(min=1-self.epsilon, max=1+self.epsilon)
            L, _ = torch.stack([r * A.detach(), clip * A.detach()]).min(0)
            v_l = A.pow(2).mean()
            L = L.mean()
            loss = -L + self.v_loss_coef * v_l
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        del self.states
            
    def end_(self, epoch, rs):
        if rs[epoch-1] >= 19 and rs[epoch] >= 19 and self.posi:
            return True

    def run(self, name="", seed=1):
        start = time.time()
        self.reset_seed(seed)
        rs = np.zeros(self.max_epochs)
        try:
            for epoch in range(self.max_epochs):
                self.update()
                rs[epoch] = self.reward_mean
                if (epoch + 1) % 10 == 0:
                    sec = int(time.time() - start)
                    h = sec // 3600
                    sec = sec % 3600
                    m = int(sec // 60)
                    sec = int(sec % 60)
                    print(epoch+1, rs[epoch], end=" time:")
                    print(h, m, sec, sep=":")
                    torch.save(self.model.to("cpu").state_dict(), "pong/ppo"+name+str(epoch+1)+".pth")
                    self.model.to(self.device)
                if self.end_(epoch, rs):
                    rs = rs[:epoch+1]
                    break
        except KeyboardInterrupt:
            rs = rs[:epoch+1]
        torch.save(self.model.to("cpu").state_dict(), "pong/ppo"+name+str(epoch+1)+".pth")
        return rs