import numpy as np
import random
import copy
from typing import List, Tuple, Optional, Dict, Any

class EnvConfig:
    MAX_HP = 40
    ATK = 6
    RADIUS = 30.0
    MAX_VELOCITY = 800.0
    MU = 0.1 #  currently using friction, mu not used
    # init position 没想好怎么办

    ARENA_SIZE = (900, 500)
    FPS = 60

    WIN_SCORE = 4
    # 还有一堆渲染设置就不写了

class TrainConfig:
    STATE_DIM:int = 24
    ACTION_DIM:int = 3
    GMM_COMPONENTS:int = 3

    CKPT_DIR = "checkpoints"
    PRETRAINED_CKPT = None  # "checkpoints/step_1000.pt"

    INITIAL_LR = 1e-4
    ALPHA_LR = 1e-4
    TOTAL_STEPS = 1e6
    GAMMA = 0.99
    TAU = 0.005

    BUFFER_SIZE:int = 1000
    BATCH_SIZE:int = 64
    MAX_STEPS:int = 100000
    SAVE_INTERVAL:int = 1000

'''
这玩意怎么写
# action = {
#     'team': 'blue',
#     'idx': seal_idx,
#     'param': (r, theta)
# }

# obs = {
#     'blue': [s.state() for s in self.blue],
#     'red': [s.state() for s in self.red],
#     'round_first': self.round_first,
#     'current_round': self.current_round,
#     'current_move_team': self.current_move_team,
#     'kills_by_blue': self.kills_by_blue,
#     'kills_by_red': self.kills_by_red,
#     'action_mask': {
#         'blue': [s.movable for s in self.blue],
#         'red': [s.movable for s in self.red],
#     }
# }

#    def state(self):
#         return {
#             'team': self.team,
#             'idx': self.idx,
#             'pos': tuple(self.pos),
#             'vel': tuple(self.vel),
#             'hp': self.hp,
#             'attack': self.attack,
#             'alive': self.alive,
#             'radius': self.radius,
#             'movable': self.movable,
#         }
'''
