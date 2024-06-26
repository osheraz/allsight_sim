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

from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("path", lambda: PATH)

log = logging.getLogger(__name__)

# Load the config YAML file from experiments/conf/allsight.yaml
@hydra.main(config_path="conf", config_name="experiment_collect_data")
def main(cfg):
    """Main program
    Args:
        cfg (config dictionary): dictionary config of the simulate scene
    """

    # start script from the path of the paraent dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # create simulator object
    simulator = Simulator(cfg=cfg)

    # create env
    simulator.create_env(cfg.allsight, obj_id=cfg.summary.indenter)

    # collect data
    simulator.collect_data(conf=cfg.summary)


if __name__ == "__main__":
    main()
