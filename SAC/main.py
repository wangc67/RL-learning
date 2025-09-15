from env import SealBattleEnv, masked_softmax_sample
import numpy as np
from agent import SACGMM
import os
from config import TrainConfig, EnvConfig


if __name__ == '__main__':
    # 集成了 人机 人人 机机 对战的main函数，注意render_mode='human'会自动加载pygame
    env = SealBattleEnv(render_mode='human')
    obs, _ = env.reset()
    terminated, truncated = False, False
    turn = 'blue'

    human_controls_blue = True # 设置为True让人类控制蓝色队伍
    human_controls_red = True  # 设置为True让人类控制红色队伍

    ckpt_file = "checkpoints/step_1000.pt"
    if ckpt_file and os.path.exists(ckpt_file) and (not human_controls_blue or not human_controls_red):
        agent = SACGMM()
        agent.load(ckpt_file)
        print(f"已加载模型 {ckpt_file}")

    while not (terminated or truncated):
        if (turn == 'blue' and human_controls_blue) or (turn == 'red' and human_controls_red):
            # 人类玩家回合
            action = env.renderer.get_human_input(obs)
            if action is None:  # 玩家关闭了窗口
                break
        else: # AI回合
            if ckpt_file is None: # 随机行动
                aa = np.random.uniform(0, 1, (3,)) 
                action_idx = masked_softmax_sample(aa, obs['action_mask'][turn])
                action = {
                    'team': turn,
                    'idx': action_idx,
                    'param': (np.random.uniform(0, 1), np.random.uniform(0, 2*np.pi))  
                }
            else: # 使用加载的模型进行推理
                action_arr = agent.select_action(obs)
                action = action_arr
            print(f"AI {turn} 选择的动作: {action}")
        obs, rwd, terminated, truncated, info = env.step(action)
        turn = env.current_move_team
        print(obs)

    print('Game over', info)
    env.close()

