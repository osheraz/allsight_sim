# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time

# import deepdish as dd
import numpy as np
import pybullet as pb
# import pybullet_data
from robot_envs.robot_openhand_env import InsertionEnv
import pybulletX as px
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import Log
# logger = logging.getLogger(__name__)


def get_forces(bodyA=None, bodyB=None, linkIndexA=None, linkIndexB=None):
    """
    get contact forces

    :return: normal force, lateral force
    """
    kwargs = {
        "bodyA": bodyA,
        "bodyB": bodyB,
        "linkIndexA": linkIndexA,
        "linkIndexB": linkIndexB,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    pts = pb.getContactPoints(**kwargs)

    totalNormalForce = 0
    totalLateralFrictionForce = [0, 0, 0]

    for pt in pts:
        totalNormalForce += pt[9]

        totalLateralFrictionForce[0] += pt[11][0] * pt[10] + pt[13][0] * pt[12]
        totalLateralFrictionForce[1] += pt[11][1] * pt[10] + pt[13][1] * pt[12]
        totalLateralFrictionForce[2] += pt[11][2] * pt[10] + pt[13][2] * pt[12]

    return totalNormalForce, totalLateralFrictionForce


# log = Log("data/grasp")

rob = InsertionEnv(robot_name='kuka', allsight_display=True)
object_start_pos = rob.get_object_pose()
pos = object_start_pos[0]

grasp_ori = (0.7120052640594666, -0.7021243464589736, 0.005861858779661195, -0.0059619353246853635)
rob.go(pos=pos, ori=grasp_ori, ori_is_quat=True, wait=True)

time_render = []
time_vis = []

dz = 0.01

t = px.utils.SimulationThread(real_time_factor=1.0)
t.start()

# t=0
gripForce = 20

normalForceList0 = []
normalForceList1 = []

print("\n")
t = 0

# rob.go(pos=pos, ori=[0,0,0])
while True:
    t += 1
    print(t)

    if t == 5:
        # Reaching
        rob.go(pos, wait=True)
    elif t == 10:
        # Grasping
        rob.grasp(gripForce=gripForce)
    elif t == 11:
        # Record sensor states
        tactileColor, tactileDepth = rob.allsights.render()
        tactileColorL, tactileColorR = tactileColor[0], tactileColor[1]
        tactileDepthL, tactileDepthR = tactileDepth[0], tactileDepth[1]

        rob.allsights.updateGUI(tactileColor, tactileDepth)

        normalForce0, lateralForce0 = get_forces(rob.robotID, rob.objID, rob.sensorLinks[0], -1)
        normalForce1, lateralForce1 = get_forces(rob.robotID, rob.objID, rob.sensorLinks[1], -1)
        normalForce = [normalForce0, normalForce1]
        # print("Normal Force Left", normalForce0, "Normal Force Right", normalForce1)
        # print("normal force", normalForce, "lateral force", lateralForce)

        objPos0, objOri0 = rob.get_object_pose()
    elif 10 < t < 15:
        # Lift
        pos[-1] += dz
        rob.go(pos)
    elif t == 15:
        # Save the data
        objPos, objOri = rob.get_object_pose()

        if objPos[2] - objPos0[2] < 60 * dz * 0.8:
            # Fail
            label = 0
        else:
            # Success
            label = 1
        # print("Success" if label == 1 else "Fail", end=" ")

        # log.save(
        #     tactileColorL,
        #     tactileColorR,
        #     tactileDepthL,
        #     tactileDepthR,
        #     gripForce,
        #     normalForce,
        #     label,
        # )
        # print("\rsample {}".format(log.id * log.batch_size + len(log.dataList)), end="")

        # print("\rsample {}".format(log.id), end="")

        # if log.id > 2000:
        #     break

        # Reset
        t = 0

        # rob.go(pos, width=0.11)
        # for i in range(100):
        #     pb.stepSimulation()

        rob.reset_robot()

        objRestartPos = [
            0.50 + 0.1 * np.random.random(),
            -0.15 + 0.3 * np.random.random(),
            0.05,
        ]
        objRestartOrientation = pb.getQuaternionFromEuler(
            [0, 0, 2 * np.pi * np.random.random()]
        )

        pos = [
            objRestartPos[0] + np.random.uniform(-0.02, 0.02),
            objRestartPos[1] + np.random.uniform(-0.02, 0.02),
            objRestartPos[2] * (1 + np.random.random() * 0.5),
        ]
        ori = [0, np.pi, 2 * np.pi * np.random.random()]
        # pos = [0.50, 0, 0.205]
        # pos = [np.random.random(0.3)]

        gripForce = 5 + np.random.random() * 15

        rob.go(pos + np.array([0, 0, 0.0]), ori=ori)
        pb.resetBasePositionAndOrientation(rob.objID, objRestartPos, objRestartOrientation)
        # for i in range(100):
        #     pb.stepSimulation()
        pb.stepSimulation()

    pb.stepSimulation()
    st = time.time()
    # color, depth = rob.allsights.render()

    time_render.append(time.time() - st)
    time_render = time_render[-100:]
    # print("render {:.4f}s".format(np.mean(time_render)), end=" ")
    st = time.time()

    # rob.allsights.updateGUI(color, depth)

    time_vis.append(time.time() - st)
    time_vis = time_vis[-100:]

    # print("visualize {:.4f}s".format(np.mean(time_vis)))

    color, depth = rob.allsights.render()
    rob.allsights.updateGUI(color, depth)
