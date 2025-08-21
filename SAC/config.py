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

            # 'current_move_team': self.current_move_team,
#     'kills_by_blue': self.kills_by_blue,
#     'kills_by_red': self.kills_by_red,
#     # action mask: shape (2, 3) booleans. True means allowed to act when that team is acting.
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