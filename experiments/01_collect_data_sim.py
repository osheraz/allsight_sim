# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from datetime import datetime
import os
import hydra
import pybullet as p
import math
import pybulletX as px
from omegaconf import DictConfig
from utils.logger import DataSimLogger

import numpy as np

from utils.geometry import rotation_matrix, concatenate_matrices
from scipy.spatial.transform import Rotation as R

# import allsight wrapper
PATH = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, PATH)
from tacto_allsight_wrapper.allsight_simulator import Simulator

log = logging.getLogger(__name__)
origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)


# Load the config YAML file from experiments/conf/allsight.yaml
@hydra.main(config_path="conf", config_name="allsight")
def main(cfg):
    """Main program
    Args:
        cfg (config dictionary): dictionary config of the simulate scene
    """

    # start script from the path of the paraent dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    save = False

    start_from = 1
    up_to = 11
    max_press_time = 10.0
    max_pressure = 4.0
    save_every_sec = 0.005

    indenter = '20'
    leds = 'rgbrgbrgb'
    gel = 'clear'
    N = 30

    conf = {'method': 'press',
            'save': save,
            'up_to': up_to,
            'start_from': start_from,
            'N': N,
            'max_press_time': max_press_time,
            'max_pressure': max_pressure,
            'leds': leds,
            'indenter': indenter,
            'gel': gel,
            'save_every_sec': save_every_sec}

    # create simulator object
    simulator = Simulator(cfg=cfg,
                          with_bg=True)

    # create env
    simulator.create_env(cfg, obj_id=indenter)

    # collect data
    simulator.collect_data(conf)


if __name__ == "__main__":
    main()
