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

# ============ 训练循环 ============ #
def train_selfplay(env, agent, cfg:TrainConfig):
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    replay_buffer = ReplayBuffer(max_size=cfg.BUFFER_SIZE)

    reward_log = []
    episode_reward = 0
    obs, _ = env.reset()

    for step in range(1, cfg.MAX_STEPS + 1):
        action = agent.select_action(obs)
        # print(step, action)
        # exit()
        # step_action = {
        #     "team": obs['current_move_team'],
        #     "idx": int(action[2]),
        #     "param": (action[0], action[1] * 2 * np.pi),  # 极坐标 (r, theta)
        # }
        step_action = action
        next_obs, reward, terminated, truncated, _ = env.step(step_action)
        agent.update_buffer(obs, action, reward, next_obs, terminated or truncated)
        episode_reward += reward
        obs = next_obs

        if terminated or truncated:
            reward_log.append(episode_reward)
            episode_reward = 0
            obs, _ = env.reset()

        agent.update()

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
    ckpt_file = None # "checkpoints/step_1000.pt"
    if ckpt_file and os.path.exists(ckpt_file):
        agent.load(ckpt_file)
        print(f"已加载模型 {ckpt_file}")

    print('training started')
    train_selfplay(env, agent, cfg=TrainConfig())
