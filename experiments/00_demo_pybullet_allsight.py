# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
import threading
import time
import hydra

import pybullet as p

import cv2
import pybulletX as px
import os
import numpy as np

log = logging.getLogger(__name__)


# import allsight wrapper
PATH = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, PATH)
from tacto_allsight_wrapper.allsight_simulator import Simulator

from omegaconf import DictConfig, OmegaConf

# Load the config YAML file from experiments/conf/allsight.yaml
@hydra.main(config_path="conf", config_name="experiment_demo")
def main(cfg):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    simulator = Simulator(cfg=cfg)
    
    simulator.create_env(cfg.allsight, obj_id=cfg.summary.indenter)
    simulator.run_sim()


if __name__ == "__main__":
    main()
