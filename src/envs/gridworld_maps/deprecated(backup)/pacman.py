import os
import numpy as np
import torch
import json
from gym.utils import seeding
from enum import IntEnum
from envs.multiagentenv import MultiAgentEnv
from envs.rendering import fill_coords, point_in_circle
from envs.utils import create_agent, create_floor, create_tile, colors

# left, right, up, down
ACTIONS = [(0, -1), (0, 1), (1, 0), (-1, 0)]
ACTIONS_NAME = ["left", "right", "up", "down", "stay"]


class PacManEnv(MultiAgentEnv):

    # Enumeration of possible actions
    class Actions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3
        stay = 4

    def __init__(self, map_name, seed=0, dense_reward=False, visualize=False):
        self.env_name = map_name
        envfile_dir = "src/envs/" + map_name.split("_")[0] + "/"
        json_file = envfile_dir + map_name + ".json"
        with open(json_file) as infile:
            args = json.load(infile)
        self.grid_orig = np.load(envfile_dir + args["grid_file"])
        self.grid = self.grid_orig.copy()
        self.img = np.load(envfile_dir + args["img_file"])
        self.height, self.width = self.grid.shape

        self.random_transition_order = False
        self.np_random, _ = seeding.np_random(seed)

        self.n_agents = args["agent_num"]
        self.starts = np.array(args["starts"])
        self.dots_ranges = np.array(args["dots"])
        self.agents = self.starts.copy()
        self.agents_pre = self.starts.copy()

        self.collide = False

        self.actions = PacManEnv.Actions

        self.action_space = []
        self.observation_space = []

        self.episode_limit = 30
        self.step_count = 0

        self.dense_reward = dense_reward
        self.visualize = visualize
        self.cur_img = None
        self.window = None

        self.total_dots = 6 * len(self.dots_ranges)
        self.reset()

    def get_state_size(self):
        # return 8
        x, y = self.grid.shape
        return (x-4) * (y-4)
    
    def get_obs_size(self):
        return 25
    
    def get_total_actions(self):
        return 5

    def reset(self):
        self.step_count = 0
        self.dots_eaten = 0
        self.dots_eaten_step = 0
        self.generate_dots()
        self.agents = self.starts.copy()
        self.agents_pre = self.starts.copy()
        for aid in range(self.n_agents):
            x, y = self.agents[aid]
            self.grid[x, y] = 3
        if self.visualize:
            self.cur_img = self.img.copy()
            self.initialize_img()
            self.update_img()

        return self.state

    def generate_dots(self):
        self.grid = self.grid_orig.copy()
        dots_poses = [self.generage_dots_poses(dots_range, 6) for dots_range in self.dots_ranges]
        for dot_pose in dots_poses:
            self.grid[dot_pose[:, 0], dot_pose[:, 1]] = 2

    def generage_dots_poses(self, dots_range, num):
        array_1 = np.arange(dots_range[0], dots_range[1])
        array_2 = np.arange(dots_range[2], dots_range[3])
        dots_poses = np.array(np.meshgrid(array_1, array_2)).T.reshape(-1, 2)
        idx = np.random.choice(np.arange(len(dots_poses)), num, replace=False)
        return dots_poses[idx]

    @property
    def state(self):
        cur_state = {
            'vec': self.get_obs(),
            'pos': self.agents.copy()
        }
        if self.visualize:
            cur_state = {
                'image': np.flip(self.cur_img, axis=0),
                'vec': self.get_obs(),
                'pos': self.agents.copy()
            }
        return cur_state

    @property
    def done(self):
        if (self.dots_eaten == self.total_dots) or (self.step_count >= self.episode_limit):
            done = True
        else:
            done = False
        return done

    def get_state(self):
        # return self.agents.flatten().copy()
        return self.grid[2:-2, 2:-2].flatten().copy()

    def get_obs(self):
        obs = []
        for agent in self.agents:
            i, j = agent
            obs.append(self.grid[i-2: i+3, j-2:j+3].flatten().copy())
        return obs

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for aid in range(self.n_agents):
            avail_agent_ids = self._available_actions(self.agents[aid])
            avail_agent = [1 if i in avail_agent_ids else 0 for i in range(5)]
            # avail_agent = [1 for i in range(5)]
            avail_actions.append(avail_agent)
        return avail_actions

    def _occupied_by_grid(self, i, j):
        if self.grid[i, j] == 1:
            return True
        return False

    def _occupied_by_dot(self, i, j):
        if self.grid[i, j] == 2:
            return True
        return False
    
    def _occupied_by_agent(self, i, j):
        if self.grid[i, j] == 3:
            return True
        return False

    # def _occupied_by_agent(self, cur_id, i, j):
    #     for aid in range(self.n_agents):
    #         if aid == cur_id:
    #             pass
    #         elif np.array_equal(self.agents[aid], [i, j]):
    #             self.collide = True
    #             return True
    #     return False

    def _available_actions(self, agent_pos):
        available_actions = set()
        available_actions.add(self.actions.stay)
        i, j = agent_pos

        assert (0 <= i <= self.height - 1) and (0 <= j <= self.width - 1), \
            'Invalid indices'

        if (i > 0) and not self._occupied_by_grid(i - 1, j):
            available_actions.add(self.actions.down)
        if (i < self.height - 1) and not self._occupied_by_grid(i + 1, j):
            available_actions.add(self.actions.up)
        if (j > 0) and not self._occupied_by_grid(i, j - 1):
            available_actions.add(self.actions.left)
        if (j < self.width - 1) and not self._occupied_by_grid(i, j + 1):
            available_actions.add(self.actions.right)

        return available_actions

    def _transition(self, actions):
        self.agents_pre = self.agents.copy()
        idx = [i for i in range(self.n_agents)]
        if self.random_transition_order:
            self.np_random.shuffle(idx)
        for aid in idx:
            action = actions[aid]
            if torch.is_tensor(action):
                action = action.item()
            if action not in self._available_actions(self.agents[aid]):
                pass
            else:
                i, j = self.agents[aid]
                if action == self.actions.up:
                    i += 1
                if action == self.actions.down:
                    i -= 1
                if action == self.actions.left:
                    j -= 1
                if action == self.actions.right:
                    j += 1
                if not self._occupied_by_agent(i, j):
                    self.agents[aid] = [i, j]
                    if self.grid[i, j] == 2:
                        self.dots_eaten += 1
                        self.dots_eaten_step += 1
                    self.grid[i, j] = 3
                    x, y = self.agents_pre[aid]
                    self.grid[x, y] = 0

                    # print("Agent {} moved {} from {} to {}".format(aid, ACTIONS_NAME[aid], self.agents_pre[aid], self.agents[aid]))
                    # print("Dots eaten: {}".format(self.dots_eaten))
                    # print("Agent observation: {}".format(self.get_obs()[aid]))

    def step(self, actions):
        # breakpoint()
        # print(actions)
        self.dots_eaten_step = 0
        self.step_count += 1
        self._transition(actions)

        if self.visualize:
            self.update_img()

        reward = self._reward()
        # reward = self.dots_eaten_step

        done = self.done

        info = {}
        if self.collide:
            info["collide"] = True
            self.collide = False

        return reward, done, info

    def _reward(self):
        rewards = 0
        if self.done:
            rewards = self.dots_eaten # * 1.0 / self.total_dots
        return rewards

    def initialize_img(self, tile_size=30):
        dots = np.where(self.grid == 2)
        for i in range(len(dots[0])):
            goal_tile = create_tile(tile_size, colors[0])
            x = dots[0][i] * tile_size
            y = dots[1][i] * tile_size
            self.cur_img[x:x + tile_size, y:y + tile_size] = goal_tile / 2

    def update_img(self, tile_size=30):
        for i in range(self.n_agents):
            x = self.agents_pre[i][0] * tile_size
            y = self.agents_pre[i][1] * tile_size
            floor_tile = create_floor(tile_size)
            self.cur_img[x:x + tile_size, y:y + tile_size] = floor_tile

        for i in range(self.n_agents):
            x = self.agents[i][0] * tile_size
            y = self.agents[i][1] * tile_size
            agent_tile = create_agent(tile_size, colors[0])
            self.cur_img[x:x + tile_size, y:y + tile_size] = agent_tile

    def render(self, mode="human"):
        if self.window is None:
            from envs import window
            self.window = window.Window('Grid World')
            self.window.show(block=False)

        if self.visualize:
            self.update_img()
        self.window.show_img(np.flip(self.cur_img, axis=0))

        return np.flip(self.cur_img, axis=0)
    
    def close(self):
        """Close StarCraft II."""
        if self.window is not None:
            self.window.close()
            self.window = None

    def get_env_info(self):
        env_info = super().get_env_info()
        return env_info
    
    def get_stats(self):
        stats = {}
        return stats