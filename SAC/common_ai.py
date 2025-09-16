from env import SealBattleEnv, masked_softmax_sample
import numpy as np
from agent import SACGMM
import os
from config import TrainConfig, EnvConfig
import random

class common_ai():
    def __init__(self):
        pass
    
    def select_action(self, obs):
        selected_seal_idx = -1
        my_team = obs['current_move_team']
        enemy_team = 'blue' if my_team == 'red' else 'red'
        for i in range(3):
            if obs['action_mask'][my_team][i] == True:
                selected_seal_idx = i
                break
        assert selected_seal_idx != -1, f"common ai cannot chosen seal idx, mask: {obs['action_mask'][my_team]}"
        rr = 1
        x0, y0 = obs[my_team][selected_seal_idx]['pos']

        crush_seal_idx = random.randint(0, 2)
        x1, y1 = obs[enemy_team][crush_seal_idx]['pos']
        theta = np.atan2((y1 - y0), (x1 - x0)) / (2 * np.pi)
        theta = theta + 1.0 if theta < 0 else theta
        return np.array([rr, theta, selected_seal_idx])

if __name__ == '__main__':
    # common ai test
    env = SealBattleEnv(render_mode='human')
    obs, _ = env.reset()
    terminated, truncated = False, False
    turn = 'blue'

    human_controls_blue = True # 设置为True让人类控制蓝色队伍
    human_controls_red = False  # 设置为True让人类控制红色队伍

    agent = common_ai()

    while not (terminated or truncated):
        if (turn == 'blue' and human_controls_blue) or (turn == 'red' and human_controls_red):
            # 人类玩家回合
            action = env.renderer.get_human_input(obs)
            if action is None:  # 玩家关闭了窗口
                break
        else: # AI回合
            if agent is None: # 随机行动
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
        # print([it['current_move_team'] for it in env.trajectory])

    print('Game over', info)
    env.close()

