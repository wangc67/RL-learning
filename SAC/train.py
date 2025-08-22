import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from env import SealBattleEnv
from config import EnvConfig, TrainConfig
from agent import SACGMM, ReplayBuffer

def process_obs(obs):
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

# ============ 训练循环 ============ #
def train_selfplay(env, agent, cfg:TrainConfig):
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    replay_buffer = ReplayBuffer(max_size=cfg.BUFFER_SIZE)

    reward_log = []
    episode_reward = 0
    obs, _ = env.reset()
    state = process_obs(obs)

    for step in range(1, cfg.MAX_STEPS + 1):
        action = agent.select_action(state)
        # print(step, action)
        # exit()
        step_action = {
            "team": obs['current_move_team'],
            "idx": int(action[2]),
            "param": (action[0], action[1] * 2 * np.pi),  # 极坐标 (r, theta)
        }
        next_obs, reward, terminated, truncated, _ = env.step(step_action)
        next_state = process_obs(next_obs)
        replay_buffer.push(state, action, reward, next_state, terminated or truncated)
        episode_reward += reward
        state = next_state
        obs = next_obs

        if terminated or truncated:
            reward_log.append(episode_reward)
            episode_reward = 0
            obs, _ = env.reset()
            state = process_obs(obs)

        if len(replay_buffer) > cfg.BUFFER_SIZE: # ????
            agent.update(replay_buffer)

        if step % 100 == 0:
            print(f"[Step {step}] 训练中...")

        if step % cfg.SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(cfg.CKPT_DIR, f"step_{step}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] 已保存 {ckpt_path}")

    plt.figure()
    plt.plot(reward_log, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.CKPT_DIR, "training_curve.png"))
    plt.show()

# ============ 使用示例 ============ #
if __name__ == "__main__":
    env = SealBattleEnv()
    agent = SACGMM(cfg=TrainConfig())

    # 如果要加载已有 checkpoint
    ckpt_file = None # "checkpoints/step_100000.pt"
    if ckpt_file and os.path.exists(ckpt_file):
        agent.load(ckpt_file)
        print(f"已加载模型 {ckpt_file}")

    print('training started')
    train_selfplay(env, agent, cfg=TrainConfig())
