import os
import matplotlib.pyplot as plt
from env import SealBattleEnv
from config import EnvConfig, TrainConfig
from agent import SACGMM, ReplayBuffer
import jsonlines
from time import time
import random
from common_ai import common_ai

# ============ 训练循环 ============ #
def train_selfplay(env, agent, cfg:TrainConfig):
    print("Training selfplay")
    time_str = str(time()).split(".")[0][-6:]
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    log_file = os.path.join(cfg.CKPT_DIR, f'reward_log_{time_str}.jsonl')

    reward_log = []
    episode_reward = 0
    obs, _ = env.reset()

    for step in range(1, cfg.MAX_STEPS + 1):
        action = agent.select_action(obs)
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
            with jsonlines.open(log_file, mode='a') as writer:
                writer.write({"step": step, "reward": episode_reward})
            episode_reward = 0
            obs, _ = env.reset()

        agent.update()

        if step % 100 == 0:
            print(f"[Step {step}] 训练中...")

        if step % cfg.SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(cfg.CKPT_DIR, f"step_{step}_{time_str}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] 已保存 {ckpt_path}")

    plt.figure()
    plt.plot(reward_log, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(cfg.CKPT_DIR, f"training_curve_self_{time_str}.png"))
    plt.show()

def train_common_ai(env, agent, opponent, cfg:TrainConfig):
    print("Training against common AI")
    time_str = str(time()).split(".")[0][-6:]
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    log_file = os.path.join(cfg.CKPT_DIR, f'reward_log_{time_str}.jsonl')

    team_choices = ['blue', 'red']

    reward_log = []
    episode_reward = 0
    obs, _ = env.reset()
    agent_team = random.choice(team_choices)
    opponent_team = 'blue' if agent_team == 'red' else 'red'

    for step in range(1, cfg.MAX_STEPS + 1):
        if obs['current_move_team'] == agent_team:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update_buffer(obs, action, reward, next_obs, terminated or truncated)
            episode_reward += reward
            obs = next_obs
        else:
            action = opponent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs

        if terminated or truncated:
            reward_log.append(episode_reward)
            with jsonlines.open(log_file, mode='a') as writer:
                writer.write({"step": step, "reward": episode_reward})
            episode_reward = 0
            obs, _ = env.reset()

        agent.update()

        if step % 100 == 0:
            print(f"[Step {step}] 训练中...")

        if step % cfg.SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(cfg.CKPT_DIR, f"step_{step}_{time_str}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] 已保存 {ckpt_path}")

    plt.figure()
    plt.plot(reward_log, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(cfg.CKPT_DIR, f"training_curve_ai_{time_str}.png"))
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

    # train_selfplay(env, agent, cfg=TrainConfig())
    train_common_ai(env, agent, opponent=common_ai(), cfg=TrainConfig())
