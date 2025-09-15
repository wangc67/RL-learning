import math
import random
import copy
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from config import EnvConfig

'''
def get_reward(obs_lst:List) -> float:
    # obs_lst: last 5 observations, including current one
    move_team = obs_lst[-2]['current_move_team'] # current_move_team是指的下一个时刻需要行动的team
    change_round = 0
    for ii in range(-1, -5, -1):
        if obs_lst[ii]['current_round'] != obs_lst[ii-1]['current_round']:
            change_round = ii
            break

    my_1_idx = -1
    if change_round == -1:
        my_0_idx = -2
        enemy_1_idx = -3
        enemy_0_idx = -5
    elif change_round == -2:
        my_0_idx = -4
        enemy_0_idx = -2
        enemy_1_idx = -3
    elif change_round == -3:
        my_0_idx = -3
        enemy_0_idx = -2
        enemy_1_idx = -5
    else:
        my_0_idx = -3
        enemy_0_idx = -2
        enemy_1_idx = -4

    my_state = {}
    enemy_state = {}
    if move_team == 'blue':
        my_state['last_hp'] = [s['hp'] for s in obs_lst[my_0_idx]['blue']]
        my_state['now_hp'] = [s['hp'] for s in obs_lst[my_1_idx]['blue']]
        my_state['last_kills'] = obs_lst[my_0_idx]['kills_by_blue']
        my_state['now_kills'] = obs_lst[my_1_idx]['kills_by_blue'] 
        enemy_state['last_hp'] = [s['hp'] for s in obs_lst[enemy_0_idx]['red']]
        enemy_state['now_hp'] = [s['hp'] for s in obs_lst[enemy_1_idx]['red']]
        enemy_state['last_kills'] = obs_lst[enemy_0_idx]['kills_by_red']
        enemy_state['now_kills'] = obs_lst[enemy_1_idx]['kills_by_red']
    else:
        my_state['last_hp'] = [s['hp'] for s in obs_lst[my_0_idx]['red']]
        my_state['now_hp'] = [s['hp'] for s in obs_lst[my_1_idx]['red']]
        my_state['last_kills'] = obs_lst[my_0_idx]['kills_by_red']
        my_state['now_kills'] = obs_lst[my_1_idx]['kills_by_red']
        enemy_state['last_hp'] = [s['hp'] for s in obs_lst[enemy_0_idx]['blue']]
        enemy_state['now_hp'] = [s['hp'] for s in obs_lst[enemy_1_idx]['blue']]
        enemy_state['last_kills'] = obs_lst[enemy_0_idx]['kills_by_blue']
        enemy_state['now_kills'] = obs_lst[enemy_1_idx]['kills_by_blue']
  
    reward = 0.0

    kd_weight_lst = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.2], [1.0, 0.7], [0.7, 0.0]]
    # change_round: 0, -1, -2, -3, -4

    kill_weight, death_weight = kd_weight_lst[change_round]
    
    reward += (my_state['now_kills'] - my_state['last_kills']) * kill_weight # 击杀奖励
    reward -= (enemy_state['now_kills'] - enemy_state['last_kills']) * death_weight # 被击杀惩罚

    if my_state['now_kills'] >= EnvConfig.WIN_SCORE:
        reward += 5.0 # 获胜奖励
    if enemy_state['now_kills'] >= EnvConfig.WIN_SCORE:
        reward -= 5.0 # 失败惩罚
    print(f'k weight: {kill_weight}, d weight: {death_weight}')
    print(f'team: {move_team}, reward: {reward}, change_round: {change_round}')
    return reward
'''

def get_reward(obs_lst:List) -> float:
    # obs_lst: last 5 observations, including current one
    move_team = obs_lst[-2]['current_move_team'] # current_move_team是指的下一个时刻需要行动的team

    my_state = {}
    enemy_state = {}
    if move_team == 'blue':
        my_state['now_hp'] = [s['hp'] for s in obs_lst[-1]['blue']]
        my_state['now_kills'] = obs_lst[-1]['kills_by_blue'] 
        enemy_state['now_hp'] = [s['hp'] for s in obs_lst[-1]['red']]
        enemy_state['now_kills'] = obs_lst[-1]['kills_by_red']
    else:
        my_state['now_hp'] = [s['hp'] for s in obs_lst[-1]['red']]
        my_state['now_kills'] = obs_lst[-1]['kills_by_red']
        enemy_state['now_hp'] = [s['hp'] for s in obs_lst[-1]['blue']]
        enemy_state['now_kills'] = obs_lst[-1]['kills_by_blue']

    reward = 0.0

    reward += (my_state['now_kills'] - enemy_state['now_kills'])* 1.0 # 击杀奖励
    reward += (sum(my_state['now_hp']) - sum(enemy_state['now_hp'])) * 0.01 # 血量差奖励
    if my_state['now_kills'] >= EnvConfig.WIN_SCORE:
        reward += 5.0 # 获胜奖励
    if enemy_state['now_kills'] >= EnvConfig.WIN_SCORE:
        reward -= 5.0 # 失败惩罚
    # print(f'team: {move_team}, reward: {reward}')
    return reward

if __name__ == '__main__':
    print("This is the reward module.")

"""
obs example:

{
'blue': [
            {
                'team': 'blue', 
                'idx': 0, 
                'pos': (156.40952117803423, 346.1272456849059), 
                'vel': (-0.0, -0.0), 
                'hp': 40, 
                'attack': 6, 
                'alive': True, 
                'radius': 30.0, 
                'movable': False
            }, 
            {
                'team': 'blue', 
                'idx': 1, 
                'pos': (100.0, 250.0), 
                'vel': (0.0, 0.0), 
                'hp': 40, 
                'attack': 6, 
                'alive': True, 
                'radius': 30.0, 
                'movable': True
            }, 
            {
                'team': 'blue', 
                'idx': 2, 
                'pos': (100.0, 375.0), 
                'vel': (0.0, 0.0), 
                'hp': 40, 'attack': 6, 
                'alive': True, 
                'radius': 30.0, 
                'movable': True
            }
        ], 
'red':  [
            {
                'team': 'red', 
                'idx': 0, 
                'pos': (800.0, 125.0), 
                'vel': (0.0, 0.0), 
                'hp': 40, 
                'attack': 6, 
                'alive': True, 
                'radius': 30.0, 
                'movable': True
            }, 
            {
                'team': 'red', 
                'idx': 1, 
                'pos': (742.5660588434687, 151.96090460755454), 
                'vel': (0.0, 0.0), 
                'hp': 28, 
                'attack': 6, 
                'alive': True, 
                'radius': 30.0, 
                'movable': True
            }, 
            {
                'team': 'red', 
                'idx': 2, 
                'pos': (800.0, 375.0), 
                'vel': (0.0, 0.0), 
                'hp': 40, 
                'attack': 6, 
                'alive': True, 
                'radius': 30.0, 
                'movable': True
            }
        ], 
'round_first': 'blue', 
'current_round': 0, 
'current_move_team': 'red', 
'kills_by_blue': 0, 
'kills_by_red': 0, 
'action_mask': {
                'blue': [False, True, True], 
                'red': [True, True, True]
               }
}
"""

