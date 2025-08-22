import math
import random
import time
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from env import SealBattleEnv
from config import TrainConfig, EnvConfig

# ============ Replay Buffer ============ #
class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )
    def __len__(self):
        return len(self.buffer)

# ============ 工具函数 ============ #
def mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, out_act=None):
    layers = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(activation())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)

# ============ Actor with GMM ============ #
class GMMActor(nn.Module):
    def __init__(self, obs_dim, action_dim, num_components=3, hidden_dims=[128,128]):
        super().__init__()
        self.num_components = num_components
        self.action_dim = action_dim
        
        state_dim = obs_dim - 3 #-----
        # shared backbone
        self.backbone = mlp(state_dim, [hidden_dims[0]], hidden_dims[1])

        # actor heads
        self.logits_head = nn.Linear(hidden_dims[1], 3) # 选择 seal idx
        self.gmm_head = nn.Linear(hidden_dims[1], 3 * 3 * 3 * 2) # mixture of 3 Gaussians, 每个有 (mean, logstd) for r,theta
        # self.gmm_logit = nn.Linear(hidden_dims[1], 9)

    def masked_softmax(self, logits, mask, temperature=1.0):
        masked_logits = logits.masked_fill(~mask, -1e9)
        prob = F.softmax(masked_logits * temperature, dim=-1)
        return prob

    def forward(self, state: torch.Tensor):
        state = state[:,:-3]
        feature = self.backbone(state)
        out = self.gmm_head(feature)
        logit = self.logits_head(feature)
        # gmm_logits = self.gmm_logit(feature)

        # logit: batch * 3(num_seals)

        """
        num of distrubutions: 3(numseals) * 3(gmm_num) * 2(action_dim)
        param for each distribution: 3, mu, std, logit

        mean: [batch * 3(num_seals) * 3(gmm_num)]
        log_std: [batch * 3(num_seals) * 3(gmm_num)]
        gmm_logits: [batch * 3(num_seals) * 3(gmm_num)]
        """

        means, log_stds, gmm_logits = torch.chunk(out, 3, dim=-1)
        means = means.view(-1, 3, 3, 2)
        means = torch.sigmoid(means) # 保证在 [0, 1] 之间
        log_stds = log_stds.view(-1, 3, 3, 2).clamp(-5,2)
        gmm_logits = gmm_logits.view(-1, 3, 3, 2)
        # print('means', means.shape, 'log_stds', log_stds.shape, 'gmm_logits', gmm_logits.shape)  # 调试输出
        return logit, means, log_stds, gmm_logits

    def gumbel_softmax(self, logits, tau=1.0, hard=False):
        eps = 1e-20
        U = torch.rand_like(logits)
        g = -torch.log(-torch.log(U + eps) + eps)
        y = logits + g
        y = F.softmax(y / tau, dim=-1)
        if hard:
            # 变成 one-hot 但梯度用 y
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            y = (y_hard - y).detach() + y
        return y

    def sample(self, state, tau=1.0):
        action_mask = (state[:, -3:] == 1)  # [B, 3]
        logit, means, log_stds, gmm_logits = self(state)
        # logit: [B,3], means/log_stds/gmm_logits: [B,3,3,2]

        # ---------- Step1: 动作选择 (soft Gumbel-Softmax) ----------
        probs_action = self.masked_softmax(logit, action_mask)  # [B,3]
        action_onehot = self.gumbel_softmax(torch.log(probs_action + 1e-20), tau=tau, hard=False)  # [B,3]

        # ---------- Step2: GMM component选择 ----------
        gmm_probs = (action_onehot.unsqueeze(-1).unsqueeze(-1) * gmm_logits).sum(dim=1)  # [B,3,2]
        gmm_probs = F.softmax(gmm_probs, dim=1)  # [B,3,2]
        comp_onehot = self.gumbel_softmax(torch.log(gmm_probs.sum(-1) + 1e-20), tau=tau, hard=False)  # [B,3]

        # ---------- Step3: mean/std计算 ----------
        means_selected = (action_onehot.unsqueeze(-1).unsqueeze(-1) * means).sum(dim=1)   # [B,3,2]
        stds_selected  = (action_onehot.unsqueeze(-1).unsqueeze(-1) * log_stds).sum(dim=1)  # [B,3,2]

        mean = (comp_onehot.unsqueeze(-1) * means_selected).sum(dim=1)  # [B,2]
        std  = torch.exp((comp_onehot.unsqueeze(-1) * stds_selected).sum(dim=1))  # [B,2]

        # ---------- Step4: 高斯 reparameterization ----------
        eps = torch.randn_like(std)
        action = mean + std * eps
        action = torch.clamp(action, 0, 1)  # [B,2]

        # ---------- Step5: 拼接动作索引 ----------
        action_idx = action_onehot.argmax(dim=-1, keepdim=True).float()  # [B,1]
        action_out = torch.cat([action, action_idx], dim=-1)  # [B,3]

        # ---------- Step6: 可导 log_prob ----------

        # 连续动作 log_prob
        normal_dist = torch.distributions.Normal(mean, std)
        log_prob_cont = normal_dist.log_prob(action[:, :2]).sum(dim=-1, keepdim=True)  # [B,1]

        # 离散动作 log_prob (soft)
        log_prob_disc = torch.sum(action_onehot * torch.log(probs_action + 1e-20), dim=-1, keepdim=True)  # [B,1]

        # GMM component log_prob (soft)
        comp_probs = gmm_probs.sum(-1)  # [B,3]
        log_prob_comp = torch.sum(comp_onehot * torch.log(comp_probs + 1e-20), dim=-1, keepdim=True)  # [B,1]

        # 总 log_prob
        log_prob = log_prob_cont + log_prob_disc + log_prob_comp  # [B,1]

        return action_out, log_prob


# ============ Q 网络 ============ #
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128,128]):
        super().__init__()
        self.net = mlp(state_dim+action_dim, hidden_dims, 1)
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

# ============ SAC-GMM Agent ============ #
class SACGMM:
    def __init__(self, cfg=TrainConfig()):
        self.actor = GMMActor(cfg.STATE_DIM, cfg.ACTION_DIM)
        self.q1 = QNetwork(cfg.STATE_DIM, cfg.ACTION_DIM)
        self.q2 = QNetwork(cfg.STATE_DIM, cfg.ACTION_DIM)
        self.target_q1 = QNetwork(cfg.STATE_DIM, cfg.ACTION_DIM)
        self.target_q2 = QNetwork(cfg.STATE_DIM, cfg.ACTION_DIM)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.INITIAL_LR)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.INITIAL_LR)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.INITIAL_LR)

        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.ALPHA_LR)

        self.gamma = cfg.GAMMA
        self.tau = cfg.TAU
        self.target_entropy = -cfg.ACTION_DIM

        self.batch_size = cfg.BATCH_SIZE

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer:ReplayBuffer):
        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next = self.target_q1(next_state, next_action)
            q2_next = self.target_q2(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob.unsqueeze(-1)
            target_q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.gamma * q_next

        q1_loss = F.mse_loss(self.q1(state, action), target_q)
        q2_loss = F.mse_loss(self.q2(state, action), target_q)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob.unsqueeze(-1) - q_new).mean()

        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau)*target_param.data)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau)*target_param.data)

        return q1_loss.item(), q2_loss.item(), actor_loss.item(), alpha_loss.item()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.item()
        }, path)

    def load(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.target_q1.load_state_dict(checkpoint["target_q1"])
        self.target_q2.load_state_dict(checkpoint["target_q2"])
        self.log_alpha = torch.tensor(checkpoint["log_alpha"], requires_grad=True)

if __name__ == '__main__':
    print('agent.py')