import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random


class Scenario(BaseScenario):
    def make_world(self, num_agents, num_targets):
        world = World()
        # set any world properties first
        world.dim_c = 0
        if num_agents == -1:
            num_agents = 3
            num_landmarks = 3
        else:
            if num_targets == -1:
                raise AssertionError("Number of targets is not assigned")
            else:
                num_landmarks = num_targets
        world.collaborative = False
        world.discrete_action = True
        world.num_agents_obs = 2
        world.num_landmarks_obs = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-world.range_p, +world.range_p, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-world.range_p, +world.range_p, world.dim_p)
            if i != 0:
                for j in range(i): 
                    while True:
                        if np.sqrt(np.sum(np.square(landmark.state.p_pos - world.landmarks[j].state.p_pos)))>0.22:
                            break
                        else: landmark.state.p_pos = np.random.uniform(-world.range_p, +world.range_p, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        # # set agent goals
        # if goals is None:
        #     goals = [i for i in range(len(world.agents))]
        #     random.shuffle(goals)
        # world.goals = goals

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            collision_dist = agent.size + l.size
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < collision_dist:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        collisions += 0.5
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        collision_dist = agent1.size + agent2.size
        return True if dist < collision_dist else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # local reward
        #dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        #rew = rew - min(dists)
        # global reward
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        # collisions penalty
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        rew -= 0.5
        return rew

    def observation(self, agent, world):
        entity_pos = []
        dist_n = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            dist_n.append(np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos))))
        # dist_sort = dist_n.copy()
        # dist_sort.sort()
        # num_landmarks_obs = world.num_landmarks_obs
        # dist_thresh = dist_sort[num_landmarks_obs-1]
        target_pos = []
        for i,pos in enumerate(entity_pos):
            if True:#dist_n[i] <= dist_thresh:
                target_pos.append(pos)
            else:
                target_pos.append(np.array([100,100]))
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        #print(target_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + target_pos)


