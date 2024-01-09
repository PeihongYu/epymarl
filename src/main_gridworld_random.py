import numpy as np
from envs.gridworld import GridWorldEnv

def main():
    map_name = "centerSquare6x6_2a"
    map_path="/nfshomes/peihong/Documents/epymarl/src/envs/gridworld_maps/maps"
    env = GridWorldEnv(map_name, map_path=map_path, render=True)

    n_actions = env.get_total_actions()
    n_agents = env.n_agents

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            # env.render()

            actions = np.random.randint(0, n_actions, n_agents)
            reward, terminated, _ = env.step(actions)
            episode_reward += reward
        
        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()
        
if __name__ == "__main__":
    main()