# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
import pybullet as p
import tacto  # Import TACTO

import pybulletX as px

# log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/allegro_hand.yaml
@hydra.main(config_path="conf", config_name="openhand")
def main(cfg):
    # Initialize allsights
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from tacto_allsight_wrapper import allsight_wrapper

    # import allsight wrapper
    PATH = os.path.join(os.path.dirname(__file__), "../")
    import sys
    sys.path.insert(0, PATH)
    import cv2
    bg = cv2.imread(os.path.join(PATH, f"experiments/conf/ref/ref_frame_rrrgggbbb.jpg"))
    conf_path = os.path.join(PATH, f"experiments/conf/sensor/config_allsight_rrrgggbbb.yml")

    allsights = allsight_wrapper.Sensor(
            **cfg.tacto, **{"config_path": conf_path},
            background=bg if True else None
        )

    # Initialize World
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Add allegro hand
    allegro = px.Body(**cfg.allegro)

    # Add cameras to tacto simulator
    allsights.add_camera(allegro.id, cfg.allsight_link_id_openhand)

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    allsights.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    panel.start()


    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    while True:
        color, depth = allsights.render()
        allsights.updateGUI(color, depth)

    t.stop()


if __name__ == "__main__":
    main()