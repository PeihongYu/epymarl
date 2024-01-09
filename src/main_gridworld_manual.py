from envs.gridworld import GridWorldEnv
from envs.utils.window import Window
import matplotlib.pyplot as plt
import numpy as np


def redraw(img):
    window.show_img(np.flip(img, axis=0))


def reset():
    env.reset()
    redraw(env.cur_img)


def step(action):
    print(action)
    reward, terminated, info = env.step(actions)
    print('step=', env._episode_steps, ', reward=', reward, ',  terminated? ', terminated, ', info: ', info)
    # print(f"agents: {env.agents}")
    # print(f"dots: {env.dots_eaten} / {env.total_dots}")

    if terminated:
        print('terminated!')
        reset()
    else:
        redraw(env.cur_img)
        # plt.imsave(f'gridworld_{env._episode_steps}.png', env.cur_img.astype(np.uint8))
        # plt.imsave(f'gridworld_img_{env._episode_steps}.png', env.img.astype(np.uint8))


def key_handler(event):
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset()
        return
    if event.key == 'left':
        actions.append(env.Actions.left)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return
    if event.key == 'right':
        actions.append(env.Actions.right)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return
    if event.key == 'up':
        actions.append(env.Actions.up)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return
    if event.key == 'down':
        actions.append(env.Actions.down)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return
    if event.key == ' ':
        actions.append(env.Actions.stay)
        if len(actions) == action_num:
            step(actions)
            actions.clear()
        return


map_name = "centerSquare6x6_4a"
map_path="Z:/Documents/epymarl/src/envs/gridworld_maps/maps"
env = GridWorldEnv(map_name, map_path=map_path, render=True)

# map_name="centerSquare6x6_2a",
# map_path="envs/gridworld_maps/maps",
# use_vec_obs=False,
# use_vec_state=False,
# sight_range=2,
# seed=None,
# random_transition_order=False,
# render=False,


actions = []
action_num = env.n_agents

window = Window('Grid World')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
