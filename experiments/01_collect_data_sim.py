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

    start_from = 11
    up_to = 50

    indenter = '25'
    leds = 'rrrgggbbb'
    gel = 'clear'

    angle_split = 20
    cyl_split = 20
    top_split = 5
    save = False

    summary = {'method': 'press',
               'save': save,
               'up_to': up_to,
               'start_from': start_from,
               'angle_split': angle_split,
               'cyl_split': cyl_split,
               'top_split': top_split,
               'leds': leds,
               'indenter': indenter,
               'gel': gel}

    # create simulator object
    simulator = Simulator(cfg=cfg,
                          summary=summary,
                          with_bg=True)

    # create env
    simulator.create_env(cfg, obj_id=indenter)

    # collect data
    simulator.collect_data(summary)


if __name__ == "__main__":
    main()
