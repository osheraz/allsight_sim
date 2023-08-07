# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time

# import deepdish as dd
import numpy as np
import pybullet as pb
import pybullet_data
from robot_openhand_env import InsertionEnv
import pybulletX as px
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# logger = logging.getLogger(__name__)

class Log:
    def __init__(self, dirName, id=0):
        self.dirName = dirName
        self.id = id
        self.dataList = []
        self.batch_size = 100
        os.makedirs(dirName, exist_ok=True)

    def save(
            self,
            tactileColorL,
            tactileColorR,
            tactileDepthL,
            tactileDepthR,
            gripForce,
            normalForce,
            label,
    ):
        data = {
            "tactileColorL": tactileColorL,
            "tactileColorR": tactileColorR,
            "tactileDepthL": tactileDepthL,
            "tactileDepthR": tactileDepthR,
            "gripForce": gripForce,
            "normalForce": normalForce,
            "label": label,
        }

        self.dataList.append(data.copy())

        if len(self.dataList) >= self.batch_size:
            id_str = "{:07d}".format(self.id)
            # os.makedirs(outputDir, exist_ok=True)
            outputDir = os.path.join(self.dirName, id_str)
            os.makedirs(outputDir, exist_ok=True)

            # print(newData["tactileColorL"][0].shape)
            newData = {k: [] for k in data.keys()}
            for d in self.dataList:
                for k in data.keys():
                    newData[k].append(d[k])

            for k in data.keys():
                fn_k = "{}_{}.h5".format(id_str, k)
                outputFn = os.path.join(outputDir, fn_k)
                # dd.io.save(outputFn, newData[k])

            self.dataList = []
            self.id += 1

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


log = Log("data/grasp")

rob = InsertionEnv(robot_name='sawyer')

rob.go(rob.pos, wait=True)

time_render = []
time_vis = []

dz = 0.005
pos = [0.50, 0, 0.19]

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
        rob.go(pos)
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
    elif 10 < t <= 15:
        # Lift
        pos[-1] += dz
        rob.go(pos)
    elif t > 30:
        # Save the data
        objPos, objOri = rob.get_object_pose()

        if objPos[2] - objPos0[2] < 60 * dz * 0.8:
            # Fail
            label = 0
        else:
            # Success
            label = 1
        # print("Success" if label == 1 else "Fail", end=" ")

        log.save(
            tactileColorL,
            tactileColorR,
            tactileDepthL,
            tactileDepthR,
            gripForce,
            normalForce,
            label,
        )
        print("\rsample {}".format(log.id * log.batch_size + len(log.dataList)), end="")

        # print("\rsample {}".format(log.id), end="")

        if log.id > 2000:
            break

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
            objRestartPos[2] * (1 + np.random.random() * 0.5) + 0.14,
        ]
        ori = [0, np.pi, 2 * np.pi * np.random.random()]
        # pos = [0.50, 0, 0.205]
        # pos = [np.random.random(0.3)]

        gripForce = 5 + np.random.random() * 15

        rob.go(pos + np.array([0, 0, 0.1]), ori=ori)
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
