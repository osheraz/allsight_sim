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


# Load the config YAML file from experiments/conf/allsight.yaml
@hydra.main(config_path="conf", config_name="allsight")
def main(cfg, blur=True, is_bg=True):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    simulator = Simulator(cfg=cfg,
                          with_bg=False)
    simulator.create_env(cfg, obj_id='20')
    simulator.run_sim()


if __name__ == "__main__":
    main()
