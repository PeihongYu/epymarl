import torch as th
import numpy as np
import json
from enum import IntEnum

from envs.multiagentenv import MultiAgentEnv
import envs.utils.utils as utils
from envs.gridworld_maps import get_map_params


# inside map array, 0 is empty, 1 is lava, 2 is wall, 3 is door, 4 is dots

class GridWorldEnv(MultiAgentEnv):

    class Actions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3
        stay = 4

    class DoorStatus(IntEnum):
        open = 0
        locked = 1

    class MapBlock(IntEnum):
        empty = 0
        lava = 1
        wall = 2
        door = 3
        dots = 4
        agent = 5

    def __init__(
            self, 
            map_name="centerSquare6x6_2a",
            map_path="envs/gridworld_maps/maps",
            use_vec_obs=False,
            use_vec_state=False,
            sight_range=2,
            seed=None,
            random_transition_order=False,
            render=False,
        ):
        self.map_name = map_name
        map_params = get_map_params(self.map_name)

        # Load the agent information
        self.map_type = map_params["map_type"]
        self.map_file = f"{map_path}/{map_params['map_file']}"

        self.n_agents = map_params["n_agents"]
        self.starts = np.array(map_params["starts"])
        self.agents = self.starts.copy()
        self.last_agents = self.starts.copy()

        if self.map_type == "goal_reaching":
            self.goals = np.array(map_params["goals"])
        if self.map_type == "apple_door":
            self.goals = np.array(map_params["goals"])
            self.door = np.array(map_params["door"])
            self.door_status = GridWorldEnv.DoorStatus.locked
        if self.map_type == "pacman":
            self.dots_ranges = np.array(map_params["dots"])

        self.collide = False
        self.step_in_lava = False

        # Set the random seed
        self._seed = seed
        self.random_transition_order = random_transition_order
        
        # Set the observation and state type
        self.use_vec_obs = use_vec_obs
        self.use_vec_state = use_vec_state
        self.sight_range = sight_range

        # Set the episode limit
        self.episode_limit = map_params["limit"]
        self._episode_steps = 0

        # Set the render flag
        self.render = render
        self.window = None

        self.reset()

    def reset(self):
        self._episode_steps = 0
        self.collide = False
        self.step_in_lava = False

        self.agents = self.starts.copy()
        self.last_agents = self.starts.copy()

        # Load the map
        self.map = np.flip(np.loadtxt(self.map_file), axis=0)
        self.height, self.width = self.map.shape
        if self.render:
            self.img = self.initialize_img()

        if self.map_type == "apple_door":
            self.update_door_status()
        if self.map_type == "pacman":
            self.generate_dots()

        if self.render:
            self.cur_img = self.update_img()

        return self.get_obs(), self.get_state()

    def get_state(self):
        if self.use_vec_state:
            return self.agents.flatten().copy()
        else:
            temp_map = self.map.copy()
            for agent in self.agents:
                temp_map[agent[0], agent[1]] = GridWorldEnv.MapBlock.agent
            return temp_map.flatten()

    def get_obs(self):
        if self.use_vec_obs:
            return [self.agents.flatten().copy() for i in range(self.n_agents)]
        else:
            obs = []
            sr = self.sight_range
            temp_map = np.pad(self.map, sr, mode="constant", constant_values=2)
            for agent in self.agents:
                temp_map[agent[0] + sr, agent[1] + sr] = GridWorldEnv.MapBlock.agent
            for agent in self.agents:
                i, j = agent
                obs.append(temp_map[i: i+sr*2+1, j:j+sr*2+1].flatten().copy())
            return obs

    def get_state_size(self):
        if self.use_vec_state:
            return self.agents.size
        else:
            return self.map.size
    
    def get_obs_size(self):
        if self.use_vec_obs:
            return self.agents.size
        else:
            return (self.sight_range * 2 + 1) ** 2
    
    def get_total_actions(self):
        # up, down, left, right, stay
        return 5
    
    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for aid in range(self.n_agents):
            cur_avail_action_set = self._available_actions(self.agents[aid])
            cur_avail_actions = [1 if i in cur_avail_action_set else 0 for i in range(5)]
            avail_actions.append(cur_avail_actions)
        return avail_actions

    ###################### pacman specific ######################
    def generate_dots(self):
        self.dots_eaten = 0
        self.total_dots = 6 * len(self.dots_ranges)
        dots_poses = [self.generage_dots_poses(dots_range, 6) for dots_range in self.dots_ranges]
        for dot_pose in dots_poses:
            self.map[dot_pose[:, 0], dot_pose[:, 1]] = 4

    def generage_dots_poses(self, dots_range, num):
        array_1 = np.arange(dots_range[0], dots_range[1])
        array_2 = np.arange(dots_range[2], dots_range[3])
        dots_poses = np.array(np.meshgrid(array_1, array_2)).T.reshape(-1, 2)
        idx = np.random.choice(np.arange(len(dots_poses)), num, replace=False)
        return dots_poses[idx]
    
    def eat_dots(self, i, j):
        if self.map[i, j] == 4:
            self.map[i, j] = 0
            self.dots_eaten += 1
    #################### end pacman specific ####################

    ###################### apple door specific ######################
    def update_door_status(self):
        # agent 1 is on its goal, which can keep the door open
        if np.array_equal(self.agents[1], self.goals[1]):
            self.open_door()
        else:
            self.close_door()

    def close_door(self):
        x, y = self.door
        self.map[x, y] = GridWorldEnv.MapBlock.door
        self.door_status = GridWorldEnv.DoorStatus.locked

    def open_door(self):
        x, y = self.door
        self.map[x, y] = GridWorldEnv.MapBlock.empty
        self.door_status = GridWorldEnv.DoorStatus.open
    #################### end apple door specific ####################

    ###################### goal reaching specific ######################
    def _occupied_by_lava(self, i, j):
        if self.map[i, j] == GridWorldEnv.MapBlock.lava:
            return True
        return False

    def check_in_lava(self, i, j):
        if self._occupied_by_lava(i, j):
            self.step_in_lava = True
    #################### end goal reaching specific ####################

    def _occupied_by_grid(self, i, j):
        if self.map[i, j] == GridWorldEnv.MapBlock.wall or self.map[i, j] == GridWorldEnv.MapBlock.door:
            return True
        return False

    def _occupied_by_agent(self, cur_id, i, j):
        for aid in range(self.n_agents):
            if aid == cur_id:
                pass
            elif np.array_equal(self.agents[aid], [i, j]):
                return True
        return False

    def _available_actions(self, agent_pos):
        available_actions = set()
        available_actions.add(GridWorldEnv.Actions.stay)
        i, j = agent_pos

        assert (0 <= i <= self.height - 1) and (0 <= j <= self.width - 1), \
            'Invalid indices'

        if (i > 0) and not self._occupied_by_grid(i - 1, j):
            available_actions.add(GridWorldEnv.Actions.down)
        if (i < self.height - 1) and not self._occupied_by_grid(i + 1, j):
            available_actions.add(GridWorldEnv.Actions.up)
        if (j > 0) and not self._occupied_by_grid(i, j - 1):
            available_actions.add(GridWorldEnv.Actions.left)
        if (j < self.width - 1) and not self._occupied_by_grid(i, j + 1):
            available_actions.add(GridWorldEnv.Actions.right)

        return available_actions

    def _transition(self, actions):
        self.last_agents = self.agents.copy()
        idx = [i for i in range(self.n_agents)]
        if self.random_transition_order:
            self.np.random.shuffle(idx)
        for aid in idx:
            self._transition_agent(aid, actions[aid])

    def _transition_agent(self, aid, action):
        if th.is_tensor(action):
            action = action.item()
        if action in self._available_actions(self.agents[aid]):
            i, j = self.agents[aid]
            if action == GridWorldEnv.Actions.up:
                i += 1
            if action == GridWorldEnv.Actions.down:
                i -= 1
            if action == GridWorldEnv.Actions.left:
                j -= 1
            if action == GridWorldEnv.Actions.right:
                j += 1
            
            if self._occupied_by_agent(aid, i, j):
                # collide with other agents, stay on the original position
                self.collide = True
            else:
                self.agents[aid] = [i, j]
                # do map specific things
                if self.map_type == "apple_door":
                    self.update_door_status()
                if self.map_type == "pacman":
                    self.eat_dots(i, j)
                if self.map_type == "goal_reaching":
                    self.check_in_lava(i, j)

    def step(self, actions):
        # tbd: check actions type
        # if not isinstance(actions, list):
        #     actions = [actions]

        self._episode_steps += 1
        self._transition(actions)
        if self.render:
            self.cur_img = self.update_img()

        reward = self.calculate_reward()
        done = self.done

        info = {"collision": False, "step_in_lava": False}
        if self.collide:
            info["collision"] = True
            self.collide = False
        if self.step_in_lava:
            info["step_in_lava"] = True
            self.step_in_lava = False

        return reward, done, info

    @property
    def done(self):
        if self._episode_steps >= self.episode_limit:
            # reaches the max steps
            return True
        if self.map_type == "goal_reaching":
            # some of the agents steps into the lava or all reach the goals
            if self.step_in_lava or np.array_equal(self.agents, self.goals):
                return True
        if self.map_type == "apple_door":
            # agent 0 reaches the goal
            if np.array_equal(self.agents[0], self.goals[0]):
                return True
        if self.map_type == "pacman":
            # all dots are eaten
            if self.dots_eaten == self.total_dots:
                return True
        return False

    def calculate_reward(self):
        if self.map_type == "goal_reaching":
            return self._calculate_goal_reaching_reward()
            
        if self.map_type == "apple_door":
            return self._calculate_apple_door_reward()
        
        if self.map_type == "pacman":
            return self._calculate_pacman_reward()

    def _calculate_goal_reaching_reward(self):
        if np.array_equal(self.agents, self.goals):
            reward = 10 - 9 * (self._episode_steps / self.episode_limit)
            return reward
        if self.collide:
            reward = -1
        return 0
    
    def _calculate_apple_door_reward(self):
        if np.array_equal(self.agents[0], self.goals[0]):
            reward = 10 - 9 * (self._episode_steps / self.episode_limit)
            return reward
        if self.collide:
            reward = -1
            return reward
        return 0
    
    def _calculate_pacman_reward(self):
        if self.done:
            reward = self.dots_eaten
            return reward

    ###################### render related functions ######################
    def initialize_img(self, tile_size=30):
        height, width = self.map.shape
        img = np.zeros([height * tile_size, width * tile_size, 3], dtype=int)
        floor_tile = utils.create_floor(tile_size)
        wall_tile = utils.create_tile(tile_size, (100, 100, 100))
        lava_tile = utils.create_lava(tile_size)

        for i in range(height):
            for j in range(width):
                x = i * tile_size
                y = j * tile_size
                if self.map[i, j] == GridWorldEnv.MapBlock.empty:
                    img[x:x + tile_size, y:y + tile_size] = floor_tile
                elif self.map[i, j] == GridWorldEnv.MapBlock.wall:
                    img[x:x + tile_size, y:y + tile_size] = wall_tile
                elif self.map[i, j] == GridWorldEnv.MapBlock.lava:
                    img[x:x + tile_size, y:y + tile_size] = lava_tile

        return img

    def update_img(self, tile_size=30):
        cur_img = self.img.copy()

        if self.map_type == "goal_reaching" or self.map_type == "apple_door":
            for i in range(len(self.goals)):
                goal_tile = utils.create_tile(tile_size, utils.agent_colors[i])
                x = self.goals[i][0] * tile_size
                y = self.goals[i][1] * tile_size
                cur_img[x:x + tile_size, y:y + tile_size] = goal_tile / 2

        # render door
        if self.map_type == "apple_door":
            x = self.door[0] * tile_size
            y = self.door[1] * tile_size
            door_tile = utils.create_door(tile_size, self.door_status)
            cur_img[x:x + tile_size, y:y + tile_size] = door_tile

        # render dots
        if self.map_type == "pacman":
            dots = np.where(self.map == GridWorldEnv.MapBlock.dots)
            for i in range(len(dots[0])):
                x = dots[0][i] * tile_size
                y = dots[1][i] * tile_size
                dots_tile = utils.create_tile(tile_size, (255, 255, 0))
                cur_img[x:x + tile_size, y:y + tile_size] = dots_tile

        # render agents
        for i in range(self.n_agents):
            x = self.agents[i][0] * tile_size
            y = self.agents[i][1] * tile_size
            agent_tile = cur_img[x:x + tile_size, y:y + tile_size]
            utils.fill_coords(agent_tile, utils.point_in_circle(0.5, 0.5, 0.31), utils.agent_colors[i])
            cur_img[x:x + tile_size, y:y + tile_size] = agent_tile

        return cur_img

    def render(self, mode="human"):
        if not self.window:
            from envs import window
            self.window = window.Window('Grid World')
            self.window.show(block=False)

        self.window.show_img(np.flip(self.cur_img, axis=0))

        return np.flip(self.cur_img, axis=0)
    #################### end render related functions ####################

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None

    def get_env_info(self):
        env_info = super().get_env_info()
        return env_info
    
    def get_stats(self):
        stats = {}
        return stats