import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from config import TrainConfig, EnvConfig

# ============ Replay Buffer ============ #
class ReplayBuffer:
    def __init__(self, max_size=2000):
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
    def __init__(self, obs_dim, action_dim, num_components=3, hidden_dims=[128,256]):
        super().__init__()
        self.num_components = num_components
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.backbone_dim = 128
        
        state_dim = obs_dim - 3 #-----
        # shared backbone
        self.backbone = mlp(input_dim=state_dim, hidden_dims=hidden_dims,
                            output_dim=self.backbone_dim, out_act=nn.ReLU)

        self.relu = nn.ReLU()
        self.seal_logit = nn.Linear(self.backbone_dim, 3)
        self.gmm_logit = nn.Linear(self.backbone_dim, 3 * self.num_components * 2) # seal 3
        self.mu_head = nn.Linear(self.backbone_dim, 3 * self.num_components * 2) # action: r, theta, idx
        self.std_head = nn.Linear(self.backbone_dim, 3 * self.num_components * 2)

    def masked_softmax(self, logits, mask, temperature=1.0):
        masked_logits = logits.masked_fill(~mask, -1e9)
        prob = F.softmax(masked_logits * temperature, dim=-1)
        return prob

    def forward(self, state: torch.Tensor):
        state = state[:,:-3]
        feature = self.backbone(state)
        
        means = self.mu_head(feature).view(-1, 3, self.num_components, 2)
        stds = self.std_head(feature).view(-1, 3, self.num_components, 2)
        gmm_logits = self.gmm_logit(feature).view(-1, 3, self.num_components, 2)
        seal_logits = self.seal_logit(feature)

        stds = self.relu(stds)
        return seal_logits, means, stds, gmm_logits

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
        seal_mask = (state[:, -3:] == 1)  # [B, 3]
        # new
        seal_logits, means, stds, gmm_logits = self(state)
        
        seal_probs = self.masked_softmax(seal_logits, seal_mask)
        seal_onehot = self.gumbel_softmax(torch.log(seal_probs+1e-20),tau=tau,hard=True)
        
        gmm_probs = (seal_onehot.unsqueeze(-1).unsqueeze(-1) * gmm_logits).sum(dim=1)  # [B,3,2]
        gmm_probs = F.softmax(gmm_probs, dim=1)  # [B,3,2]
        gmm_onehot = self.gumbel_softmax(torch.log(gmm_probs.sum(-1) + 1e-20), tau=tau, hard=True)  # [B,3]
        # print(f'seal_onehot: {seal_onehot}, gmm_onehot: {gmm_onehot}')
        means_selected = (seal_onehot.unsqueeze(-1).unsqueeze(-1) * means).sum(dim=1)   # [B,3,2]
        stds_selected  = (seal_onehot.unsqueeze(-1).unsqueeze(-1) * stds).sum(dim=1)  # [B,3,2]
        # print(f'mean_selected: {means_selected}, stds_selected: {stds_selected}')
        mean = (gmm_onehot.unsqueeze(-1) * means_selected).sum(dim=1)  # [B,2]
        std  = (gmm_onehot.unsqueeze(-1) * stds_selected).sum(dim=1) + 1e-5  # [B,2]
        # print(f'mean: {mean}, std: {std}')
        eps = torch.randn_like(std)
        action = mean + std * eps # [B, 2]
        # print(action)
        # print(action[:, 0:1])
        action = torch.cat([torch.sigmoid(action[:, 0:1]), torch.frac(action[:, 1:])], dim=-1)
        # print(action.shape)
        # print(action)
        # exit()

        action_seal_idx = seal_onehot.argmax(dim=-1, keepdim=True).float() # [B, 1]
        action_total = torch.cat([action, action_seal_idx], dim=-1) # [B, 3]


        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("Actor输出NaN! 输入state:", mean, std, state)
	    # 连续动作 log_prob
        normal_dist = torch.distributions.Normal(mean, std)
        log_prob_cont = normal_dist.log_prob(action).sum(dim=-1, keepdim=True)  # [B,1]

        # 离散动作 log_prob (soft)
        log_prob_disc = torch.sum(seal_onehot * torch.log(seal_probs + 1e-20), dim=-1, keepdim=True)  # [B,1]

        # GMM component log_prob (soft)
        comp_probs = gmm_probs.sum(-1)  # [B,3]
        log_prob_comp = torch.sum(gmm_onehot * torch.log(comp_probs + 1e-20), dim=-1, keepdim=True)  # [B,1]

        # 总 log_prob
        log_prob = log_prob_cont + log_prob_disc + log_prob_comp  # [B,1]
        # print(f'action: {action_total}, log_prob: {log_prob}')
        return action_total, log_prob

# ============ Q 网络 ============ #
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128,128]):
        super().__init__()
        self.net = mlp(state_dim+action_dim, hidden_dims, 1)
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

# ============ SAC-GMM Agent ============ #
class SACGMM:
    def __init__(self, cfg=TrainConfig(), mode='train'):
        self.mode = mode # train or test, if test, no need to load q nets
        self.cfg = cfg
        self.actor = GMMActor(cfg.STATE_DIM, cfg.ACTION_DIM)
        
        self.gamma = cfg.GAMMA
        self.tau = cfg.TAU
        self.target_entropy = -cfg.ACTION_DIM
        self.batch_size = cfg.BATCH_SIZE

        if self.mode == 'train':
            self.reset_train(self.cfg)

    def reset_train(self, cfg:TrainConfig):
        self.replay_buffer = ReplayBuffer(max_size=cfg.BUFFER_SIZE)
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

    def select_action(self, obs, eval_mode=False):
        # 这里可以eval用mle，train用采样，tbd
        state = self.process_obs(obs)
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.cfg.BATCH_SIZE: # 其实不太懂这个条件写啥
            return 
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

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

    def process_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        # state: 42
        # (x,y,hp) * 3, kill, (x,y,hp) * 3, kill, xian_shou, action_mask
        def process_state(s):
            tmp = []
            for item in s:
                x, y = item['pos']
                hp = item['hp']
                tmp.extend([x, y, hp])
            return np.array(tmp, dtype=float)

        current_move_team = obs['current_move_team']
        round_first = [1] if obs['round_first'] == current_move_team else [0]
        blue = process_state(obs['blue'])
        red = process_state(obs['red'])
        kills_by_blue = obs['kills_by_blue']
        kills_by_red = obs['kills_by_red']
        action_mask = obs['action_mask']['blue'] if current_move_team == 'blue' else obs['action_mask']['red']

        first_team = blue if current_move_team == 'blue' else red
        second_team = red if current_move_team == 'blue' else blue
        first_kill = [kills_by_blue if current_move_team == 'blue' else kills_by_red]
        second_kill = [kills_by_red if current_move_team == 'blue' else kills_by_blue]
        state = [*first_team, *first_kill, *second_team, *second_kill, *round_first, *action_mask]
        return np.array(state)

    def update_buffer(self, obs, action, reward, next_obs, done):
        state = self.process_obs(obs)
        next_state = self.process_obs(next_obs)
        self.replay_buffer.push(state, action, reward, next_state, done)

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
        if self.mode == 'train':
            self.q1.load_state_dict(checkpoint["q1"])
            self.q2.load_state_dict(checkpoint["q2"])
            self.target_q1.load_state_dict(checkpoint["target_q1"])
            self.target_q2.load_state_dict(checkpoint["target_q2"])
            self.log_alpha = torch.tensor(checkpoint["log_alpha"], requires_grad=True)

if __name__ == '__main__':
    print('agent.py')
    # 为啥都是0？
    cfg = TrainConfig()
    agent = SACGMM(cfg=cfg, mode='train')
    agent.load('checkpoints/step_1000.pt', device='cuda:0')
    time.sleep(2)
    mm = torch.cuda.memory_allocated()
    print(f'GPU memory allocated: {mm:.2f} MB')
    del agent
    time.sleep(2)
    mm = torch.cuda.memory_allocated()
    print(f'GPU memory allocated after del: {mm:.2f} MB')
    agent = SACGMM(cfg=cfg, mode='test')
    agent.load('checkpoints/step_1000.pt', device='cuda:0')
    time.sleep(2)
    mm = torch.cuda.memory_allocated()
    print(f'GPU memory allocated after test agent created: {mm:.2f} MB')

    exit()
