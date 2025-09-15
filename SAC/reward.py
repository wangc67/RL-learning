import math
import random
import copy
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from config import EnvConfig

def get_reward(obs_lst:List) -> float:
    # obs_lst: last 3 observations, including current one
    # 分别是 我方，对方，我方 三次obs
    last_obs = obs_lst[0]
    obs = obs_lst[-1]
    move_team = obs_lst[1]['current_move_team'] # current_move_team是指的下一个时刻需要行动的team
    my_state = {}
    enemy_state = {}
    if move_team == 'blue':
        my_state['last_hp'] = [s['hp'] for s in last_obs['blue']]
        my_state['now_hp'] = [s['hp'] for s in obs['blue']]
        my_state['last_kills'] = last_obs['kills_by_blue']
        my_state['now_kills'] = obs['kills_by_blue'] 
        enemy_state['last_hp'] = [s['hp'] for s in last_obs['red']]
        enemy_state['now_hp'] = [s['hp'] for s in obs['red']]
        enemy_state['last_kills'] = last_obs['kills_by_red']
        enemy_state['now_kills'] = obs['kills_by_red']
    else:
        my_state['last_hp'] = [s['hp'] for s in last_obs['red']]
        my_state['now_hp'] = [s['hp'] for s in obs['red']]
        my_state['last_kills'] = last_obs['kills_by_red']
        my_state['now_kills'] = obs['kills_by_red'] 
        enemy_state['last_hp'] = [s['hp'] for s in last_obs['blue']]
        enemy_state['now_hp'] = [s['hp'] for s in obs['blue']]
        enemy_state['last_kills'] = last_obs['kills_by_blue']
        enemy_state['now_kills'] = obs['kills_by_blue']
    reward = 0.0
    reward += (my_state['now_kills'] - my_state['last_kills']) * 1.0 # 击杀奖励
    reward -= (enemy_state['now_kills'] - enemy_state['last_kills']) * 1.0 # 被击杀惩罚
    # reward -= sum([my_state['last_hp'][i] - my_state['now_hp'][i] for i in range(len(my_state['now_hp']))]) * 1.0 # 受到伤害惩罚
    # reward += sum([enemy_state['last_hp'][i] - enemy_state['now_hp'][i] for i in range(len(enemy_state['now_hp']))]) * 1.0 # 造成伤害奖励
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

