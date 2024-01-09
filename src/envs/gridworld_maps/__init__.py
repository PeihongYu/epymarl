from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from envs.gridworld_maps import gridworld_maps


def get_map_params(map_name):
    map_param_registry = gridworld_maps.get_gridworld_map_registry()
    return map_param_registry[map_name]