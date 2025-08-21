"""
这个是目前最好用的

创死时候，尸体会继续完成碰撞，都停下来再重生

"""

import math
import random
import copy
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

class Seal:
    def __init__(self, team: str, idx: int, pos: Tuple[float,float], attack: float, max_hp: float, radius: float=30.0):
        self.team = team  # 'blue' or 'red'
        self.idx = idx
        self.pos = list(pos)
        self.vel = [0.0, 0.0]
        self.attack = attack
        self.max_hp = max_hp
        self.hp = max_hp
        self.radius = radius
        self.alive = True
        self.respawns = 0
        self.movable = True

    def state(self):
        return {
            'team': self.team,
            'idx': self.idx,
            'pos': tuple(self.pos),
            'vel': tuple(self.vel),
            'hp': self.hp,
            'attack': self.attack,
            'alive': self.alive,
            'radius': self.radius,
            'movable': self.movable,
        }

class SealBattleEnv:
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 60}

    def __init__(self,
                 arena_size=(900, 500),
                 seed: Optional[int]=None,
                 friction_per_second: float = 200.0,  # linear speed reduction per second
                 physics_dt: float = 1/60.0, # 要和 metadata.render_fps 匹配
                 max_shot_speed: float = 800.0,
                 seals_per_team: int = 3,
                 render_mode: Optional[str] = None):
        
        self.arena_size = arena_size
        self.width, self.height = arena_size
        self.rng = random.Random(seed)
        self.friction_per_second = friction_per_second
        self.dt = physics_dt
        self.max_shot_speed = max_shot_speed
        self.seals_per_team = seals_per_team
        self.render_mode = render_mode # -------

        # configured initial positions (centered on each half)
        self._blue_init_positions = [(100, self.height*(i+1)/(self.seals_per_team+1)) for i in range(self.seals_per_team)]
        self._red_init_positions = [(self.width-100, self.height*(i+1)/(self.seals_per_team+1)) for i in range(self.seals_per_team)]
        self.blue: List[Seal] = []
        self.red: List[Seal] = []

        self.round_first = 'blue'  # who starts the next round
        self.current_move_team = self.round_first
        self.current_round = 0
        self.kills_by_blue = 0  # counts red respawns
        self.kills_by_red = 0
        self.max_enemy_respawns_to_win = 3 #--------------------------------------

        self._build_seals()

        # storage for last-generated frames (for renderer convenience)
        self._last_frames = []

    def _build_seals(self):
        fixed_hp = 40.0
        fixed_atk = 15.0
        self.blue = [Seal('blue', i, self._blue_init_positions[i], attack=fixed_atk, max_hp=fixed_hp) for i in range(self.seals_per_team)]
        self.red  = [Seal('red', i, self._red_init_positions[i], attack=fixed_atk, max_hp=fixed_hp) for i in range(self.seals_per_team)]

    def seed(self, s:int): # api for gym 
        self.rng.seed(s)

    def reset(self) -> Dict[str,Any]: # api for gym
        self._build_seals()
        self.round_first = 'blue'
        self.current_move_team = self.round_first
        self.current_round = 0
        self.kills_by_blue = 0
        self.kills_by_red = 0
        self._last_frames = []

        observation = self._get_obs()

        if self.render_mode == 'human':
            if not hasattr(self, 'renderer'):
                self.renderer = SealBattleRenderer(self.arena_size)
            self.renderer.play_frames([self._snapshot()], obs=observation, fps=self.metadata['render_fps'])
        return observation, None

    def close(self): # api for gym
        if hasattr(self, 'renderer'):
            self.renderer.close()
    
    def _get_obs(self) -> Dict[str,Any]: # observation format
        obs = {
            'blue': [s.state() for s in self.blue],
            'red': [s.state() for s in self.red],
            'round_first': self.round_first,
            'current_round': self.current_round,
            'current_move_team': self.current_move_team,
            'kills_by_blue': self.kills_by_blue,
            'kills_by_red': self.kills_by_red,
            # action mask: shape (2, 3) booleans. True means allowed to act when that team is acting.
            'action_mask': {
                'blue': [s.movable for s in self.blue],
                'red': [s.movable for s in self.red],
            }
        }
        return obs

    def _apply_friction(self, vel, dt):
        vx, vy = vel
        speed = math.hypot(vx, vy)
        if speed <= 0: return [0.0, 0.0]
        decel = self.friction_per_second * dt
        new_speed = max(0.0, speed - decel)
        factor = new_speed / speed if speed>0 else 0
        return [vx*factor, vy*factor]

    def _wall_collision(self, s:Seal):
        if s.pos[0] - s.radius < 0:
            s.pos[0] = s.radius
            s.vel[0] = -s.vel[0]
        if s.pos[0] + s.radius > self.width:
            s.pos[0] = self.width - s.radius
            s.vel[0] = -s.vel[0]
        if s.pos[1] - s.radius < 0:
            s.pos[1] = s.radius
            s.vel[1] = -s.vel[1]
        if s.pos[1] + s.radius > self.height:
            s.pos[1] = self.height - s.radius
            s.vel[1] = -s.vel[1]

    def _seal_collision(self, a:Seal, b:Seal):
        dx, dy = b.pos[0]-a.pos[0], b.pos[1]-a.pos[1]
        dist = math.hypot(dx, dy)
        min_dist = a.radius + b.radius
        # 方向单位向量
        nx, ny = dx/dist, dy/dist if dist>0 else (1.0,0.0)
        # 重叠分离
        overlap = (min_dist - dist)/2.0
        a.pos[0] -= nx*overlap
        a.pos[1] -= ny*overlap
        b.pos[0] += nx*overlap
        b.pos[1] += ny*overlap

        # 弹性碰撞速度更新（二维, 质量相等）
        dvx, dvy = a.vel[0]-b.vel[0], a.vel[1]-b.vel[1]
        dot = dvx*nx + dvy*ny
        if dot > 0:
            a.vel[0] -= dot*nx
            a.vel[1] -= dot*ny
            b.vel[0] += dot*nx
            b.vel[1] += dot*ny

    def _step_physics_for_frames(self)->List[Tuple[Seal, Seal]]:
        # Move seals, handle wall bounces, detect collisions pairwise and apply damage
        # frames_out: will append snapshots after each physics sub-step
        # This physics step resolves positions and collisions but NOT respawns until end of full action sequence
        all_seals = self.blue + self.red
        # move
        for s in all_seals:
            s.pos[0] += s.vel[0]*self.dt
            s.pos[1] += s.vel[1]*self.dt
            s.vel = self._apply_friction(s.vel, self.dt)
            # wall collisions: reflect velocity
            self._wall_collision(s)

        # collisions: elastic + single damage per pair
        damaged_pairs = list()

        for i in range(len(all_seals)):
            for j in range(i+1, len(all_seals)):
                a, b = all_seals[i], all_seals[j]
                dx, dy = b.pos[0]-a.pos[0], b.pos[1]-a.pos[1]
                dist = math.hypot(dx, dy)
                min_dist = a.radius + b.radius
                if dist < min_dist - 1e-6:
                    self._seal_collision(a, b)  # elastic collision logic

                    damaged_pairs.append((a, b))  # 记录已损伤的海豹对

        return damaged_pairs  # 返回所有造成伤害的海豹对
        
    def _calc_damage(self, damaged_pairs: List[Tuple[Seal, Seal]], team:str):
        for a, b in damaged_pairs:
            if (a.alive and b.alive) and (a.team != b.team):
                if a.team != team:
                    a.hp -= b.attack
                else:
                    b.hp -= a.attack

    def _snapshot(self):
        return {
            'blue': [copy.deepcopy(s.state()) for s in self.blue],
            'red': [copy.deepcopy(s.state()) for s in self.red],
        }

    def generate_action_frames(self, team: str, seal_idx: int, action: Optional[Tuple[float,float]]) -> List[Dict[str,Any]]:
        """
        Return a list of frame snapshots produced by simulating a single seal launch.
        Does NOT handle respawn counting; it will apply HP reductions in collisions.
        """
        frames = []
        acting_list = self.blue if team=='blue' else self.red
        if action is None:
            # no-op: still return one snapshot
            frames.append(self._snapshot())
            return frames
        r, theta = action
        speed = max(0.0, min(1.0, r)) * self.max_shot_speed
        vx = speed * math.cos(theta)
        vy = speed * math.sin(theta)
        acting_seal = acting_list[seal_idx]
        # assign velocity to that seal
        acting_seal.vel[0] += vx
        acting_seal.vel[1] += vy

        # simulate physics for this launched seal until its speed falls below threshold
        t = 0.0
        max_sim_time = 10.0
        while t < max_sim_time:
            damaged_pairs = self._step_physics_for_frames() # -------------------------------
            self._calc_damage(damaged_pairs, team)
            # append frame snapshot
            frames.append(self._snapshot())
            # stop when the launched seal (and also others) have near-zero speed
            all_moving = any(math.hypot(s.vel[0], s.vel[1])>1.0 for s in (self.blue + self.red))
            t += self.dt
            if not all_moving:
                break
        # store last frames for renderer convenience
        self._last_frames = frames
        return frames

    def _respawn_if_needed(self):
        # After finishing collisions for an action (or full round), process deaths: seals with hp <= 0
        # Death rule: "海豹死亡时，应该先完成碰撞，再死亡。如果重生位置与其他海豹重合则错开位置。"
        for s in (self.blue + self.red):
            if s.hp <= 0:
                # complete collisions already done; now respawn to mother team's initial pos
                if s.team=='blue':
                    base_pos = list(self._blue_init_positions[s.idx])
                else:
                    base_pos = list(self._red_init_positions[s.idx])
                # avoid overlap: if overlap with any alive seal, shift randomly small amount until not overlapping
                others = [o for o in (self.blue + self.red) if o is not s]
                safe_pos = base_pos
                offset = 0
                while any(math.hypot(safe_pos[0]-o.pos[0], safe_pos[1]-o.pos[1]) < (s.radius+o.radius-1e-6) for o in others) and offset < 20:
                    safe_pos[0] += (offset%2*2-1)* (s.radius*1.1)
                    safe_pos[1] += ((offset//2)%2*2-1) * (s.radius*1.1)
                    offset += 1
                s.pos = safe_pos
                s.vel = [0.0, 0.0]
                s.hp = s.max_hp
                s.alive = True
                s.respawns += 1
                # update kill counters
                if s.team=='red':
                    # blue caused this respawn (per rules: enemy respawn counts for friendly kills)
                    self.kills_by_blue += 1
                else:
                    self.kills_by_red += 1

    def step(self, action: Dict): # api for gym
        """
        Each action ai is either (r,theta) where r in [0,1] scaled to max_shot_speed, theta in radians, or None.
        action = {'team': 'blue' or 'red', 'idx': seal_idx, 'param': (r, theta)}
        """
        assert action['team'] == self.current_move_team, f"Expected team {self.current_move_team} to act, got {action['team']}"
        acting_list = self.blue if action['team']=='blue' else self.red
        assert acting_list[action['idx']].movable, f"Seal {action['idx']} of team {action['team']} is not allowed to act this round"

        frames_all = []

        print(f'{self.current_round} round, {self.current_move_team} team acting')

        # simulate action, get frames
        frames = self.generate_action_frames(action['team'], action['idx'], action['param'])
        frames_all.extend(frames)
        # after each action, process respawns (ensuring collisions completed before death)
        self._respawn_if_needed()
        acting_list[action['idx']].movable = False  # mark this seal as having acted this round
        self.current_move_team = 'red' if self.current_move_team == 'blue' else 'blue'  # switch to the other team
        
        # after full team round, advance round counter and alternate round_first
        dont_reset_round = False
        for s in (self.blue + self.red):
            dont_reset_round = dont_reset_round or s.movable
        if not dont_reset_round:
            self.current_round += 1
            self.round_first = 'red' if self.round_first=='blue' else 'blue'
            self.current_move_team = self.round_first  # reset current move team to the new round's first team
            for s in (self.blue + self.red):    
                s.movable = True  # reset movable flag for next 
                
        print(f'next move: {self.current_move_team}')

        # store last frames
        self._last_frames = frames_all

        info = {
            'kills_by_blue': self.kills_by_blue,
            'kills_by_red': self.kills_by_red,
        }

        observation = self._get_obs()

        reward = self._get_reward() # tbd -------------------------------------------------

        terminated = (self.kills_by_blue >= self.max_enemy_respawns_to_win) or (self.kills_by_red >= self.max_enemy_respawns_to_win)
        truncated = False  # 没有时间限制导致的结束
        if self.render_mode == 'human':
            self.renderer.play_frames(self._last_frames, obs=observation, fps=self.metadata['render_fps'])

        return observation, reward, terminated, truncated, info

    def get_last_frames(self) -> List[Dict[str,Any]]:
        return self._last_frames

    def _get_reward(self):
        return 0.0

class SealBattleRenderer:
    def __init__(self, arena_size=(900,500), scale:float=1.0, caption: str = 'SealBattle'):
        # pygame import is local to keep core env free of pygame dependency
        import pygame
        pygame.init()
        self.pygame = pygame
        self.arena_size = arena_size
        self.status_bar_height = 30
        window_size = (arena_size[0], arena_size[1] + self.status_bar_height)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.bg_color = (30, 160, 220)
        self.font = pygame.font.SysFont('Arial', 16)
        
        # 添加鼠标交互状态
        self.dragging = False
        self.selected_seal = None
        self.drag_start_pos = None
        self.drag_current_pos = None

    def render_snapshot(self, snapshot, obs):
        # single snapshot draw
        pygame = self.pygame
        self.screen.fill((50, 50, 50), (0, 0, self.arena_size[0], self.status_bar_height))
        self.screen.fill(self.bg_color, (0, self.status_bar_height, self.arena_size[0], self.arena_size[1]))
        kills_text = self.font.render(f"Blue Kills: {obs['kills_by_blue']} | Red Kills: {obs['kills_by_red']}", 
                                    True, (255, 255, 255))
        self.screen.blit(kills_text, (10, 5))
        
        # 绘制回合信息
        turn_text = self.font.render(f"Current Turn: {obs['current_move_team']}, Current Round: {obs['current_round']}", True, (255, 255, 255))
        self.screen.blit(turn_text, (300, 5))
        
        for s in snapshot['blue']:
            x,y = int(s['pos'][0]), int(s['pos'][1]) + self.status_bar_height
            r = int(s['radius'])
            color = (50,120,255) if s['alive'] else (120,120,120)
            pygame.draw.circle(self.screen, color, (x,y), r)
            # outline swim ring
            pygame.draw.circle(self.screen, (0,0,0), (x,y), r, 2)

            hp_text = self.font.render(f"{int(s['hp'])}+{s['movable']}", True, (255,255,255)) 
            text_width, text_height = self.font.size(f"{int(s['hp'])}+{s['movable']}") 
            self.screen.blit(hp_text, (x - text_width // 2, y - text_height // 2)) 
            # 如果这个海豹被选中，绘制选中指示器
            if self.selected_seal and self.selected_seal['team'] == 'blue' and self.selected_seal['idx'] == s['idx']:
                pygame.draw.circle(self.screen, (255, 255, 0), (x, y), r+5, 2)
                
        for s in snapshot['red']:
            x,y = int(s['pos'][0]), int(s['pos'][1]) + self.status_bar_height
            r = int(s['radius'])
            color = (240,80,80) if s['alive'] else (120,120,120)
            pygame.draw.circle(self.screen, color, (x,y), r)
            pygame.draw.circle(self.screen, (0,0,0), (x,y), r, 2)
            hp_text = self.font.render(f"{int(s['hp'])}+{s['movable']}", True, (255,255,255)) 
            text_width, text_height = self.font.size(f"{int(s['hp'])}+{s['movable']}") 
            self.screen.blit(hp_text, (x - text_width // 2, y - text_height // 2)) 
            # 如果这个海豹被选中，绘制选中指示器
            if self.selected_seal and self.selected_seal['team'] == 'red' and self.selected_seal['idx'] == s['idx']:
                pygame.draw.circle(self.screen, (255, 255, 0), (x, y), r+5, 2)

        # 如果正在拖拽，绘制拖拽指示器
        if self.dragging and self.selected_seal and self.drag_current_pos:
            seal_pos = self.selected_seal['pos']
            seal_screen_pos = (seal_pos[0], seal_pos[1] + self.status_bar_height)
            mouse_pos = self.drag_current_pos
            
            # 绘制从海豹到鼠标的线
            pygame.draw.line(self.screen, (255, 255, 0), seal_screen_pos, mouse_pos, 2)
            
            # 计算距离（速度指示）
            dx = mouse_pos[0] - seal_screen_pos[0]
            dy = mouse_pos[1] - seal_screen_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            max_indicator_radius = 100  # 最大指示器半径
            normalized_speed = min(1.0, distance / max_indicator_radius)
            
            # 绘制速度指示器（圆形）
            indicator_radius = int(normalized_speed * max_indicator_radius)
            pygame.draw.circle(self.screen, (255, 255, 0), seal_screen_pos, indicator_radius, 2)
            
            # 显示速度百分比
            speed_text = self.font.render(f"{int(normalized_speed * 100)}%", True, (255, 255, 0))
            self.screen.blit(speed_text, (seal_screen_pos[0] + indicator_radius + 5, seal_screen_pos[1]))

        pygame.display.flip()

    def play_frames(self, frames: List[Dict[str,Any]], obs, fps: int = 60):
        pygame = self.pygame
        running = True
        for snapshot in frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break
            self.render_snapshot(snapshot, obs)
            self.clock.tick(fps)
        pygame.time.wait(500)
        return

    def get_human_input(self, obs):
        """获取人类玩家的输入"""
        pygame = self.pygame
        self.dragging = False
        self.selected_seal = None
        self.drag_start_pos = None
        self.drag_current_pos = None
        
        current_team = obs['current_move_team']
        action_mask = obs['action_mask'][current_team]
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                    
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 鼠标按下，检查是否点击了可行动的海豹
                    mouse_pos = event.pos
                    mouse_x, mouse_y = mouse_pos
                    mouse_y -= self.status_bar_height  # 调整到游戏区域坐标
                    
                    # 检查是否点击了当前队伍的海豹
                    seals = obs['blue'] if current_team == 'blue' else obs['red']
                    for i, seal in enumerate(seals):
                        if not action_mask[i] or not seal['alive']:
                            continue
                            
                        seal_x, seal_y = seal['pos']
                        distance = math.sqrt((mouse_x - seal_x)**2 + (mouse_y - seal_y)**2)
                        
                        if distance <= seal['radius']:
                            self.selected_seal = {'team': current_team, 'idx': i, 'pos': seal['pos']}
                            self.drag_start_pos = mouse_pos
                            self.drag_current_pos = mouse_pos
                            self.dragging = True
                            break
                
                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    # 鼠标移动，更新当前位置
                    self.drag_current_pos = event.pos
                
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
                    # 鼠标释放，计算行动参数
                    mouse_pos = event.pos
                    seal_pos = self.selected_seal['pos']
                    seal_screen_pos = (seal_pos[0], seal_pos[1] + self.status_bar_height)
                    
                    # 计算向量（从海豹到鼠标）
                    dx = mouse_pos[0] - seal_screen_pos[0]
                    dy = mouse_pos[1] - seal_screen_pos[1]
                    
                    # 计算距离（速度）
                    distance = math.sqrt(dx*dx + dy*dy)
                    max_indicator_radius = 100
                    normalized_speed = min(1.0, distance / max_indicator_radius)
                    
                    # 计算角度（弧度）
                    angle = math.atan2(dy, dx)
                    # 转换为反方向（海豹被推出去的方向）
                    angle += math.pi
                    # 规范化到0-2π范围
                    angle %= 2 * math.pi
                    
                    # 保存行动参数
                    action = {
                        'team': current_team,
                        'idx': self.selected_seal['idx'],
                        'param': (normalized_speed, angle)
                    }
                    
                    # 重置拖拽状态并渲染一帧以清除指示器
                    self.dragging = False
                    self.selected_seal = None
                    self.drag_start_pos = None
                    self.drag_current_pos = None
                    
                    # 渲染一帧以清除指示器
                    self.render_snapshot(obs, obs)
                    pygame.display.flip()
                    
                    return action
            
            # 渲染当前状态（包括拖拽指示器）
            self.render_snapshot(obs, obs)
            self.clock.tick(60)
            
        return None

    def close(self):
        try:
            self.pygame.quit()
        except Exception:
            pass

def masked_softmax_sample(array, mask):
    # assert array.shape == mask.shape, "Array and mask must have the same shape"
    masked_array = np.where(mask, array, -np.inf)
    exp_values = np.exp(masked_array - np.max(masked_array))  # 数值稳定性处理, no temperature
    probabilities = exp_values / np.sum(exp_values)
    
    flat_probs = probabilities.flatten()
    flat_index = np.random.choice(len(flat_probs), p=flat_probs)
    
    selected_index = np.unravel_index(flat_index, array.shape)
    return int(selected_index[0])

if __name__ == '__main__':
    env = SealBattleEnv(render_mode='human')
    obs, _ = env.reset()
    terminated = False
    truncated = False
    turn = 'blue'

    while not (terminated or truncated):
        # generate random actions for 3 seals
        action = {
            'team': turn,
            'idx': 0,
            'param': (np.random.uniform(0, 1), np.random.uniform(0, 2*np.pi))  
        }
        aa = np.random.uniform(0, 1, (3,)) 
        action['idx'] = masked_softmax_sample(aa, obs['action_mask'][turn])
        # print(action)

        obs, rwd, terminated, truncated, info = env.step(action)
        # frames = env.get_last_frames()
        # turn = 'red' if turn == 'blue' else 'blue'
        turn = env.current_move_team # ---------------------------------

    print('Game over', info)
    env.close()

