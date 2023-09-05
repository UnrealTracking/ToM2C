import os
import json
import math
import torch
import random
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from datetime import datetime

from MSMTC.DigitalPose2D.render import render
from model import A3C_Single
from utils import goal_id_filter
#from main import parser
import random 

#args = parser.parse_args()


class Pose_Env_Base:
    def __init__(self, reset_type, args,
                 nav='Goal',  # Random, Goal
                 config_path="PoseEnvLarge_multi.json",
                 setting_path=None,
                 ):
        self.nav = nav
        self.reset_type = reset_type
        self.ENV_PATH = 'MSMTC/DigitalPose2D'
        if setting_path:
            self.SETTING_PATH = setting_path
        else:
            self.SETTING_PATH = os.path.join(self.ENV_PATH, config_path)
        with open(self.SETTING_PATH, encoding='utf-8') as f:
            setting = json.load(f)

        self.env_name = setting['env_name']

        if args.num_agents == -1:
            self.n = setting['cam_number']
            self.num_target = setting['target_number']
        else:
            self.n = args.num_agents
            self.num_target = args.num_targets
        
        self.cam_area = np.array(setting['cam_area'])
        self.reset_area = setting['reset_area']
        # for obstacle
        self.obstacle_pos_list = None
        if self.reset_type == 3:
            #self.obstacle_numRange = setting['obstacle_numRange']
            self.obstacle_numRange = [self.n-1,self.n]
            self.num_obstacle = np.random.randint(self.obstacle_numRange[0], self.obstacle_numRange[1])
            self.obstacle_radiusRange = setting['obstacle_radius']
            self.obstacle_radius_list = [0 for i in range(self.num_obstacle)]
            self.obstacle_reset_area = setting['obstacle_reset_area']

        # for executor
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions_player = setting['continous_actions_player']

        self.max_steps = setting['max_steps']

        self.visual_distance = setting['visual_distance']
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 1000)

        # define action space for coordinator
        self.action_space = [spaces.Discrete(2) for i in range(self.n * self.num_target)]
        self.rotation_scale = setting['rotation_scale']

        # define observation space
        self.state_dim = 4
        if self.reset_type != 3:
            self.observation_space = np.zeros((self.n, self.num_target, self.state_dim), int)
        else:
            self.observation_space = np.zeros((self.n, self.num_target+self.num_obstacle, self.state_dim), int)

        # render 
        self.render_save = args.render_save
        self.render = args.render

        # communication edges for render
        self.comm_edges = None

        self.cam = dict()
        for i in range(self.n):
            self.cam[i] = dict(
                location=[0, 0],
                rotation=[0],
            )

        self.count_steps = 0
        self.goal_keep = 0
        self.KEEP = 10
        self.goals4cam = np.ones([self.n, self.num_target])

        # construct target_agent
        self.target_type_prob = [0.3, 0.7]
        if 'Goal' in self.nav:
            self.random_agents = [GoalNavAgent(i, self.continous_actions_player, self.cam, self.visual_distance, self.reset_area)
                                  for i in range(self.num_target)]

        self.slave_rule = (args.load_executor_dir is None)

        if not self.slave_rule:
            self.device = torch.device('cpu')
            self.slave = A3C_Single(np.zeros((1, 1, 4)), [spaces.Discrete(3)], args)
            self.slave = self.slave.to(self.device)
            saved_state = torch.load(
                args.load_executor_dir,  # 'trainedModel/best_executor.pth',
                map_location=lambda storage, loc: storage)
            self.slave.load_state_dict(saved_state['model'], strict=False)
            self.slave.eval()

    def set_location(self, cam_id, loc):
        self.cam[cam_id]['location'] = loc

    def get_location(self, cam_id):
        return self.cam[cam_id]['location']

    def set_rotation(self, cam_id, rot):
        for i in range(len(rot)):
            if rot[i] > 180:
                rot[i] -= 360
            if rot[i] < -180:
                rot[i] += 360
        self.cam[cam_id]['rotation'] = rot

    def get_rotation(self, cam_id):
        return self.cam[cam_id]['rotation']
    
    def get_cam_states(self):
        cam_states=None
        for i in range(self.n):
            rotation=self.get_rotation(i)
            location=self.get_location(i)
            state=np.array(location + rotation)
            if cam_states is None:
                cam_states=np.expand_dims(state,0)
            else:
                state=np.expand_dims(state,0)
                cam_states=np.concatenate((cam_states,state),0)
        return cam_states

    def get_hori_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[2]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now

    def get_distance(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        d = np.sqrt(y_delt * y_delt + x_delt * x_delt)
        return d
    
    def target_init_sample(self):
        # let target be initially located in the tracking area of one camera
        cam = np.random.randint(0, self.n)
        cam_loc = self.get_location(cam)
        cam_rot = self.get_rotation(cam)
        theta = (cam_rot[0] + np.random.randint(-45, 46)) * math.pi/180
        distance = np.random.randint(100, self.visual_distance - 100)
        x = cam_loc[0] + distance * math.cos(theta)
        y = cam_loc[1] + distance * math.sin(theta)
        return [float(x),float(y)]
        #return [float(np.random.randint(self.start_area[0], self.start_area[1])),float(np.random.randint(self.start_area[2], self.start_area[3]))]
    def obstacle_init_sample(self, loc_a, loc_b):
        d = self.get_distance(loc_a, loc_b)
        R = self.visual_distance
        #r1 = math.sqrt(R**2 - (d/2)**2)
        r2 = R - d/2
        #r = min(np.random.rand()*(r1+r2)/2, r2)
        r= np.random.rand() * r2 * 0.8
        c_x = (loc_a[0]+loc_b[0])/2
        c_y = (loc_a[1]+loc_b[1])/2
        theta = np.random.randint(0,360) * math.pi/180
        x = c_x + r * math.cos(theta)
        y = c_y + r * math.sin(theta)
        return [x,y]

    def get_mask(self):
        # agents only communicate with other agents who overlap with itself
        # this value should be updated in reset
        return self.mask
    
    def reset_mask(self):
        mask_all = []
        for i in range(self.n):
            mask = []
            loc_i = self.get_location(i)
            for j in range(self.n):
                if i == j:
                    mask.append(0)
                else:
                    loc_j = self.get_location(j)
                    if self.get_distance(loc_i, loc_j) > 2 * self.visual_distance:
                        mask.append(0)
                    else:
                        mask.append(1)
            mask_all.append(mask)
        mask_all = np.array(mask_all)
        return mask_all
    
    def reset(self):

        # reset camera
        camera_id_list = [i for i in range(self.n)]
        #random.shuffle(camera_id_list)

        for i in range(self.n):
            cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                       np.random.randint(self.cam_area[i][2], self.cam_area[i][3])
                       ]
            self.set_location(camera_id_list[i], cam_loc)  # shuffle

        # reset mask
        self.mask = self.reset_mask()

        for i in range(self.n):
            cam_rot = self.get_rotation(i)

            # start with non-focusing
            angle_h = np.random.randint(-180, 180)
            cam_rot[0] += angle_h * 1.0

            self.set_rotation(i, cam_rot)

        # reset the position and shape of obstacles
        if self.reset_type==3:
            self.num_obstacle = np.random.randint(self.obstacle_numRange[0], self.obstacle_numRange[1])
            self.obstacle_pos_list = []
            #choices = [(i, (i+1)%self.n) for i in range(self.n)]
            choices = []
            for i in range(self.n-1):
                choices.append((i,i+1))
            '''
            for i in range(self.n):
                for j in range(i+1,self.n):
                    choices.append((i,j))
            '''
            choices = random.sample(choices, self.num_obstacle)
            for k in range(self.num_obstacle):
                loc_a = self.get_location(choices[k][0])
                loc_b = self.get_location(choices[k][1])
                #dist = self.get_distance(loc_a, loc_b)
                for t in range(20):
                    #tmp = [float(np.random.randint(self.obstacle_reset_area[0], self.obstacle_reset_area[1])),
                    #       float(np.random.randint(self.obstacle_reset_area[2], self.obstacle_reset_area[3]))]
                    tmp_loc = self.obstacle_init_sample(loc_a, loc_b)
                    tmp_R = np.random.randint(self.obstacle_radiusRange[0], self.obstacle_radiusRange[1])
                    flag = True
                    for i in range(self.n):
                        if self.get_distance(self.get_location(i), tmp_loc) < tmp_R:
                            flag = False
                            break
                    if flag:
                        break
                if t == 20:
                    print("obstacle reset tried for 20 times")
                '''
                flag = True
                while flag:
                    tmp = [float(np.random.randint(self.obstacle_reset_area[0], self.obstacle_reset_area[1])),
                           float(np.random.randint(self.obstacle_reset_area[2], self.obstacle_reset_area[3]))]
                    flag = False
                    for i in range(self.n):
                        if self.get_distance(self.get_location(i), tmp) < self.obstacle_radius:
                            flag = True  #
                '''
                self.obstacle_pos_list.append(tmp_loc)
                self.obstacle_radius_list[k] = tmp_R
        # reset targets
        self.target_pos_list = []
        for _ in range(self.num_target):
            # tmp = [float(np.random.randint(self.start_area[0], self.start_area[1])),
            #        float(np.random.randint(self.start_area[2], self.start_area[3]))]
            tmp = self.target_init_sample()
            if self.reset_type == 3:
                for i in range(20):
                    if self.collision(tmp[0], tmp[1]):
                        tmp = self.target_init_sample()
                        #print("resample target initial state")
                    else:
                        break
                if i == 20:
                    print("obstacle reset tried for 20 times")
            self.target_pos_list.append(tmp)
        self.target_pos_list = np.array(self.target_pos_list)
        # reset agent
        p = self.target_type_prob
        choices = [0, 1] # 0 for random walk, 1 for goal nav
        self.target_type = np.random.choice(choices, size=self.num_target, p=p)
        for i in range(len(self.random_agents)):
            if 'Goal' in self.nav:
                self.random_agents[i].reset(self.cam)

        self.count_steps = 0
        self.goal_keep = 0
        self.goals4cam = np.ones([self.n, self.num_target])

        info = dict(
            Done=False,
            Reward=[0 for i in range(self.n)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
        )

        gt_directions = []
        gt_distance = []
        cam_info = []
        for i in range(self.n):
            # for target navigation
            cam_loc = self.get_location(i)
            cam_rot = self.get_rotation(i)
            cam_info.append([cam_loc, cam_rot])
            gt_directions.append([])
            gt_distance.append([])
            for j in range(self.num_target):
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                gt_directions[i].append([angle_h])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                gt_distance[i].append(d)

            info['Cam_Pose'].append(cam_loc + cam_rot)

        info['Directions'] = np.array(gt_directions)
        info['Distance'] = np.array(gt_distance)
        info['Target_Pose'] = np.array(self.target_pos_list)  # copy.deepcopy
        info['Reward'], info['Global_reward'], others = self.multi_reward(cam_info, self.goals4cam)
        if others:
            info['Camera_target_dict'] = self.Camera_target_dict = others['Camera_target_dict']
            info['Target_camera_dict'] = self.Target_camera_dict = others['Target_camera_dict']

        state, self.state_dim = self.preprocess_pose(info)
        return state

    def visible(self, cam_i, target_j, distance, angle):
        # whether target is visible from cam_i
        if self.reset_type == 2:
            return distance <= self.visual_distance #and 1-abs(angle)/45>0
        if self.reset_type == 3:
            if not distance <= self.visual_distance: #and 1-abs(angle)/45>0)
                return False
            # whether the obstacle block the view
            cam_loc = self.get_location(cam_i)
            cam_rot = self.get_rotation(cam_i)
            direction = list(cam_rot+angle)
            for i in range(self.num_obstacle):
                position_obstacle = self.obstacle_pos_list[i]
                angle_ = self.get_hori_direction(cam_loc+direction, position_obstacle)
                obstacle_distance = self.get_distance(cam_loc, position_obstacle)
                distance_vertical = math.sin(abs(angle_)/180*math.pi)*obstacle_distance
                if 0<abs(angle_)<90 and distance_vertical<self.obstacle_radius_list[i] and obstacle_distance<distance:
                    return False
            return True

    def collision(self, target_x, target_y):
        # target not go into the obstacle
        #distance = float('inf')
        for i in range(self.num_obstacle):
            distance = np.linalg.norm([self.obstacle_pos_list[i][0] - target_x, self.obstacle_pos_list[i][1] - target_y])
            if distance <= self.obstacle_radius_list[i]:
                return True
            #distance = min(distance, np.linalg.norm([self.obstacle_pos_list[i][0] - target_x, self.obstacle_pos_list[i][1] - target_y]))
        return False

    def target_move(self):
        step = 10
        if 'Random' in self.nav:
            for i in range(self.num_target):
                self.target_pos_list[i][:3] += [np.random.randint(-1 * step, step),
                                                np.random.randint(-1 * step, step)]
        elif 'Goal' in self.nav:
            delta_time = 0.13
            for i in range(self.num_target):  # only one
                loc = list(self.target_pos_list[i])

                if self.target_type[i] == 0: # random walk target
                    for _ in range(20):
                        delta_x = np.random.randint(-1 * step, step)
                        delta_y = np.random.randint(-1 * step, step)
                        if self.reset_type == 3 and self.collision(loc[0]+delta_x, loc[1]+delta_y):
                            continue
                        else:
                            break
                    self.target_pos_list[i][:2] += [delta_x ,delta_y]
                    continue

                # not random walk target
                action = self.random_agents[i].act(loc)

                target_hpr_now = np.array(action[1:])
                delta_x = target_hpr_now[0] * action[0] * delta_time
                delta_y = target_hpr_now[1] * action[0] * delta_time
                for _ in range(20):
                    if self.reset_type == 3 and self.collision(loc[0] + delta_x, loc[1] + delta_y):
                        # collide with an obstacle, so sample a new goal
                        #self.random_agents[i].generate_goal(self.random_agents[i].goal_area)
                        delta_x = 0
                        delta_y = 0
                        #print("collided")
                        break

                    if loc[0] + delta_x < self.reset_area[0] or loc[0] + delta_x > self.reset_area[1] or \
                        loc[1] + delta_y < self.reset_area[2] or loc[1] + delta_y > self.reset_area[3]: #or \
                        #self.reset_type == 3 and self.collision(loc[0] + delta_x, loc[1] + delta_y):
                        action = self.random_agents[i].act(loc)

                        target_hpr_now = np.array(action[1:])
                        delta_x = target_hpr_now[0] * action[0] * delta_time
                        delta_y = target_hpr_now[1] * action[0] * delta_time
                    else:
                        break
                if _ == 20:
                    print("Target action Sample for 20 times")
                self.target_pos_list[i][0] += delta_x
                self.target_pos_list[i][1] += delta_y

    def step(self, actions, obstacle=False):
        # obstacle: whether the action contains obstacle or not
        info = dict(
            Done=False,
            Reward=[0 for i in range(self.n)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps
        )

        if not obstacle:
            self.goals4cam = np.squeeze(actions) # num_agents * (num_targets)
        else:
            actions = np.squeeze(actions)  # num_agents * (num_targets + num_obstacles)
            self.goals4cam = actions[:,:self.num_target]
            obstacle_goals = actions[:,self.num_target:]
        # target move
        self.target_move()

        # camera move
        cam_info = []
        for i in range(self.n):
            cam_loc = self.get_location(i)
            cam_rot = self.get_rotation(i)
            cam_info.append([cam_loc, cam_rot])
        
        if not obstacle:
            r, gr, others, cam_info = self.simulate(self.goals4cam, cam_info, keep=10)
        else:
            r, gr, others, cam_info = self.simulate(self.goals4cam, cam_info, keep=10, obstacle_goals=obstacle_goals)
        
        for i in range(self.n):
            cam_loc, cam_rot = cam_info[i]
            self.set_rotation(i, cam_rot)

        if others:
            info['Coverage_rate'] = others['Coverage_rate']
            info['Camera_target_dict'] = self.Camera_target_dict = others['Camera_target_dict']
            info['Target_camera_dict'] = self.Target_camera_dict = others['Target_camera_dict']
            info['Camera_local_goal'] = others['Camera_local_goal']
            info['cost'] = others['cost']

        info['Reward'] = np.array(r)
        info['Global_reward'] = np.array(gr)

        gt_directions = []
        gt_distance = []
        for i in range(self.n):
            # for target navigation
            cam_loc = self.get_location(i)
            cam_rot = self.get_rotation(i)
            gt_directions.append([])
            gt_distance.append([])
            for j in range(self.num_target):
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                gt_directions[i].append([angle_h])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                gt_distance[i].append(d)

            info['Cam_Pose'].append(self.get_location(i) + self.get_rotation(i))

        info['Target_Pose'] = np.array(self.target_pos_list)  # copy.deepcopy
        info['Distance'] = np.array(gt_distance)
        info['Directions'] = np.array(gt_directions)

        self.count_steps += 1
        # set your done condition
        if self.count_steps >= self.max_steps:
            info['Done'] = True

        reward = info['Global_reward']

        state, self.state_dim = self.preprocess_pose(info, GoalMap=self.goals4cam)
        return state, reward, info['Done'], info

    def get_baseline_action(self, cam_loc_rot, goals, i, obstacle_goals=None):
        camera_target_visible = []
        for k, v in self.Camera_target_dict.items():
            camera_target_visible += v

        goal_ids = goal_id_filter(goals)
        if len(goal_ids) != 0:
            if self.slave_rule:
                if obstacle_goals is None:
                    target_position = (self.target_pos_list[goal_ids]).mean(axis=0)  # avg pos: [x,y,z]
                else:
                    obstacle_ids = goal_id_filter(obstacle_goals)
                    selected_targets = np.array(self.target_pos_list[goal_ids])
                    selected_obstacles = (np.array(self.obstacle_pos_list))[obstacle_ids]
                    all_goals = np.concatenate((selected_targets,selected_obstacles),axis=0)
                    target_position = (all_goals).mean(axis=0)
                
                angle_h = self.get_hori_direction(cam_loc_rot, target_position)

                action_h = angle_h // self.rotation_scale
                action_h = np.clip(action_h, -1, 1)
                action_h *= self.rotation_scale
                action = [action_h]
            else:
                tmp = []
                for j in range(len(self.target_pos_list[goal_ids])):
                    tar_p = self.target_pos_list[goal_ids][j]
                    angle_h = self.get_hori_direction(cam_loc_rot, tar_p)
                    d = self.get_distance(cam_loc_rot, tar_p)
                    tmp.append([i / 4, j / 5, angle_h / 180, d / 2000])
                target = np.zeros((1, self.num_target, 4))
                target[0, :len(tmp)] = tmp
                values, actions, entropies, log_probs = self.slave(torch.from_numpy(target).float().to(self.device),
                                                                   test=True)
                action = actions.item()
                action = np.array(self.discrete_actions[action]) * self.rotation_scale
        else:
            action = np.array(
                self.discrete_actions[np.random.choice(range(len(self.discrete_actions)))]) * self.rotation_scale

        return action

    def simulate(self, GoalMap, cam_info, keep=-1, obstacle_goals=None):
        cost = 0
        gre = np.array([0.0])
        for _ in range(keep):
            # camera move
            visible = []
            Cam_Pose = []
            for i in range(self.n):
                cam_loc, cam_rot = cam_info[i]
                if obstacle_goals is None:
                    action = self.get_baseline_action(cam_loc + cam_rot, GoalMap[i], i)
                else:
                    # obstacle_goals: num_agents * num_obstacles
                    action = self.get_baseline_action(cam_loc + cam_rot, GoalMap[i], i, obstacle_goals[i])
                if action[0] != 0:
                    cost += 1
                cam_rot[0] += action[0]
                cam_info[i] = cam_loc, cam_rot
                Cam_Pose.append(cam_loc + cam_rot)
                sub_visible = []
                for j in range(self.num_target):
                    angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                    d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                    sub_visible.append(self.visible(i, j, d, angle_h))
                visible.append(sub_visible)

            # target move
            self.target_move()
            #
            r, gr, others = self.multi_reward(cam_info, GoalMap)
            gre += gr

            # render
            if self.render:
                render(Cam_Pose, np.array(self.target_pos_list), goal=self.goals4cam, obstacle_pos=np.array(self.obstacle_pos_list), comm_edges=self.comm_edges, obstacle_radius=self.obstacle_radius_list, save=self.render_save, visible=visible)

        cost = cost / keep
        others['cost'] = cost

        return r, gre / keep, others, cam_info

    def close(self):
        pass

    def seed(self, para):
        pass

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area

    def angle_reward(self, angle_h, d):
        hori_reward = 1 - abs(angle_h) / 45.0
        visible = hori_reward > 0 and d <= self.visual_distance
        if visible:
            reward = np.clip(hori_reward, -1, 1)  # * (self.visual_distance-d)
        else:
            reward = -1
        return reward, visible

    def simplified_multi_reward(self, cam_info):
        coverage_rate = []
        min_angle = [180 for j in range(self.num_target)]
        for i in range(self.n):
            cam_loc, cam_rot = cam_info[i]
            for j in range(self.num_target):
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                if d < self.visual_distance + 100:
                    # 
                    min_angle[j] = min(min_angle[j], np.abs(angle_h))
                reward, visible = self.angle_reward(angle_h, d)
                if visible:
                    coverage_rate.append(j)
        min_angle_sum = sum(min_angle)
        coverage_rate = len(set(coverage_rate)) / self.num_target
        return coverage_rate, min_angle_sum
        
    def multi_reward(self, cam_info, goals4cam):
        camera_local_rewards = []
        camera_local_goal = []

        camera_target_dict = {}
        target_camera_dict = {}
        captured_targets = []
        coverage_rate = []
        for i in range(self.n):
            cam_loc, cam_rot = cam_info[i]
            camera_target_dict[i] = []
            local_rewards = []
            captured_num = 0
            goal_num = 0
            for j in range(self.num_target):
                if not target_camera_dict.get(j):
                    target_camera_dict[j] = []
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                reward, visible = self.angle_reward(angle_h, d)
                #reward = self.angle_reward(angle_h, d)
                #visible = self.visible(i,j,d,angle_h)
                if visible:
                    camera_target_dict[i].append(j)
                    target_camera_dict[j].append(i)
                    coverage_rate.append(j)
                    if goals4cam is None or goals4cam[i][j] > 0:
                        captured_targets.append(j)
                        captured_num += 1

                if goals4cam is None and visible or goals4cam is not None and goals4cam[i][j] > 0:
                    local_rewards.append(reward)
                    goal_num += 1
            camera_local_goal.append(captured_num / goal_num if goal_num != 0 else -1)
            camera_local_rewards.append(np.mean(local_rewards) if len(local_rewards) > 0 else 0)
            camera_local = camera_local_rewards

        # real coverage rate
        coverage_rate = len(set(coverage_rate)) / self.num_target

        camera_global_reward = [coverage_rate]

        # if torch.is_tensor(goals4cam):
        #     goals_sum = torch.sum(goals4cam)
        # else:
        #     goals_sum = np.sum(goals4cam)
        # if goals_sum == 0:
        #     camera_global_reward = [-0.1]
        if len(set(captured_targets)) == 0:
           camera_global_reward = [-0.1]

        return camera_local, camera_global_reward, {'Camera_target_dict': camera_target_dict,
                                                    'Target_camera_dict': target_camera_dict,
                                                    'Coverage_rate': coverage_rate,
                                                    'Captured_targetsN': len(set(captured_targets)),
                                                    'Camera_local_goal': camera_local_goal
                                                    }

    def preprocess_pose(self, info, GoalMap=None):
        cam_pose_info = np.array(info['Cam_Pose'])
        target_pose_info = np.array(info['Target_Pose'])
        angles = info['Directions']
        distances = info['Distance']

        camera_num = len(cam_pose_info)
        target_num = len(target_pose_info)

        # normalize center
        center = np.mean(cam_pose_info[:, :2], axis=0)
        cam_pose_info[:, :2] -= center
        if target_pose_info is not None:
            target_pose_info[:, :2] -= center

        # scale
        norm_d = int(max(np.linalg.norm(cam_pose_info[:, :2], axis=1, ord=2))) + 1e-8
        cam_pose_info[:, :2] /= norm_d
        if target_pose_info is not None:
            target_pose_info[:, :2] /= norm_d

        state_dim = 4
        feature_dim = target_num * state_dim if self.reset_type!=3 else (target_num+self.num_obstacle)*state_dim
        state = np.zeros((camera_num, feature_dim))
        for cam_i in range(camera_num):
            # target info
            target_info = []
            for target_j in range(target_num):
                if self.reset_type == 1 or self.reset_type >= 2 and self.visible(cam_i, target_j, distances[cam_i, target_j], angles[cam_i, target_j]):
                    angle_h = angles[cam_i, target_j]
                    target_info += [cam_i / camera_num, target_j / target_num, angle_h / 180, distances[cam_i, target_j] / 2000]  # 2000 is related with the area of cameras
                else:
                    target_info += [0,0,0,0]
            if self.reset_type==3:
                for obstacle_i in range(self.num_obstacle):
                    cam_loc = self.get_location(cam_i)
                    cam_rot = self.get_rotation(cam_i)
                    obstacle_angle = self.get_hori_direction(cam_loc + cam_rot, self.obstacle_pos_list[obstacle_i])
                    obstacle_distance = self.get_distance(cam_loc + cam_rot, self.obstacle_pos_list[obstacle_i])
                    #visible = 1-abs(obstacle_angle)/45>=0 and obstacle_distance-self.obstacle_radius<=self.visual_distance or \
                    #          0<abs(obstacle_angle)-45<90 and math.sin((abs(obstacle_angle)-45)/180*math.pi)*obstacle_distance<self.obstacle_radius and obstacle_distance-self.obstacle_radius<self.visual_distance
                    visible = obstacle_distance <= self.visual_distance
                    if visible:  # obstacle is visible from cam_i
                        target_info += [cam_i / camera_num, obstacle_i / self.num_obstacle, obstacle_angle / 180, obstacle_distance / 2000]
                    else:
                        target_info += [0, 0, 0, 0]
            assert len(target_info) == feature_dim
            state[cam_i] = target_info
        if self.reset_type!=3:
            state = state.reshape((camera_num, target_num, state_dim))
        else:
            state = state.reshape((camera_num, target_num+self.num_obstacle, state_dim))

        return state, state_dim


class GoalNavAgent(object):

    def __init__(self, id, action_space, cam, visual_distance, goal_area, goal_list=None):
        self.id = id
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.cam_radius = visual_distance
        self.goal_area = self.cam2goal_area(cam)
        self.goal_list = goal_list
        self.goal = self.generate_goal(self.goal_area)

        self.max_len = 200

    def cam2goal_area(self, cam):
        #area = [cam[i]['location'] for i in range(len(cam))]
        #return area
        x = np.array([cam[i]['location'][0] for i in range(len(cam))])
        y = np.array([cam[i]['location'][1] for i in range(len(cam))])
        x_max = np.max(x) + self.cam_radius
        x_min = np.min(x) - self.cam_radius
        y_max = np.max(y) + self.cam_radius
        y_min = np.min(y) - self.cam_radius
        area = [x_min, x_max, y_min, y_max]
        return area

    def act(self, pose):
        self.step_counter += 1
        if len(self.pose_last[0]) == 0:
            self.pose_last[0] = np.array(pose)
            self.pose_last[1] = np.array(pose)
            d_moved = 30
        else:
            d_moved = min(np.linalg.norm(np.array(self.pose_last[0]) - np.array(pose)),
                          np.linalg.norm(np.array(self.pose_last[1]) - np.array(pose)))
            self.pose_last[0] = np.array(self.pose_last[1])
            self.pose_last[1] = np.array(pose)
        if self.check_reach(self.goal, pose) or d_moved < 10 or self.step_counter > self.max_len:
            self.goal = self.generate_goal(self.goal_area)
            for _ in range(20):
                if np.linalg.norm(self.goal[:2] - pose[:2]) == 0:
                    self.goal = self.generate_goal(self.goal_area)
                    print("resample target goal")
                else:
                    break
            if _ == 20:
                print("Target Goal sample for 20 times")
            self.velocity = np.random.randint(self.velocity_low, self.velocity_high)

            self.step_counter = 0

        if np.linalg.norm(self.goal[:2] - pose[:2]) == 0:
            print("target already reached goal.{}".format(self.goal[:2] - pose[:2]))
        assert np.linalg.norm(self.goal[:2] - pose[:2]) != 0
        delt_unit = (self.goal[:2] - pose[:2]) / np.linalg.norm(self.goal[:2] - pose[:2])
        velocity = self.velocity * (1 + 0.2 * np.random.random())
        return [velocity, delt_unit[0], delt_unit[1]]

    def reset(self,cam):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal_area = self.cam2goal_area(cam)
        self.goal = self.generate_goal(self.goal_area)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = [[], []]

    def generate_goal(self, goal_area):
        if self.goal_list and len(self.goal_list) != 0:
            index = self.goal_id % len(self.goal_list)
            goal = np.array(self.goal_list[index])
        else:
            '''
            cam_num = len(goal_area)
            radius = self.cam_radius
            cam_id = np.random.randint(0,cam_num)
            theta = 2*np.random.rand()*math.pi
            x = goal_area[cam_id][0] + radius * math.cos(theta)
            y = goal_area[cam_id][1] + radius * math.sin(theta)
            '''
            x = np.random.randint(goal_area[0], goal_area[1])
            y = np.random.randint(goal_area[2], goal_area[3])
            goal = np.array([x, y])
        self.goal_id += 1
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 5