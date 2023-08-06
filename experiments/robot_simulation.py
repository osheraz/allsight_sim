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
import tacto  # import TACTO
from robot import Robot
import pybulletX as px

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, cameraResolution=None):
        if cameraResolution is None:
            cameraResolution = [480, 480]

        self.cameraResolution = cameraResolution

        camTargetPos = [0.5, 0, 0.05]
        camDistance = 0.4
        upAxisIndex = 2

        yaw = 90
        pitch = -30.0
        roll = 0

        fov = 60
        nearPlane = 0.01
        farPlane = 100

        self.viewMatrix = pb.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = cameraResolution[0] / cameraResolution[1]

        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )

    def get_image(self):
        img_arr = pb.getCameraImage(
            self.cameraResolution[0],
            self.cameraResolution[1],
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2]  # color data RGB
        dep = img_arr[3]  # depth data
        return rgb, dep


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
            visionColor,
            visionDepth,
            gripForce,
            normalForce,
            label,
    ):
        data = {
            "tactileColorL": tactileColorL,
            "tactileColorR": tactileColorR,
            "tactileDepthL": tactileDepthL,
            "tactileDepthR": tactileDepthR,
            "visionColor": visionColor,
            "visionDepth": visionDepth,
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


def get_object_pose():
    res = pb.getBasePositionAndOrientation(objID)

    world_positions = res[0]
    world_orientations = res[1]

    if (world_positions[0] ** 2 + world_positions[1] ** 2) > 0.8 ** 2:
        pb.resetBasePositionAndOrientation(objID, objStartPos, objStartOrientation)
        return objStartPos, objStartOrientation

    world_positions = np.array(world_positions)
    world_orientations = np.array(world_orientations)

    return (world_positions, world_orientations)


log = Log("data/grasp")

# digits = tacto.Sensor(width=224, height=224, visualize_gui=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tacto_allsight_wrapper import allsight_wrapper
import sys
import cv2

# import allsight wrapper
PATH = os.path.join(os.path.dirname(__file__), "../")
sys.path.insert(0, PATH)

bg = cv2.imread(os.path.join(PATH, f"experiments/conf/ref/ref_frame_rrrgggbbb.jpg"))
conf_path = os.path.join(PATH, f"experiments/conf/sensor/config_allsight_rrrgggbbb.yml")

allsights = allsight_wrapper.Sensor(
    width=224, height=224, visualize_gui=True, **{"config_path": conf_path},
    background=bg if True else None
)

# Initialize World
px.init()
logging.info("Initializing world")
physicsClient = pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
pb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth

pb.resetDebugVisualizerCamera(
    cameraDistance=0.6,
    cameraYaw=15,
    cameraPitch=-20,
    # cameraTargetPosition=[-1.20, 0.69, -0.77],
    cameraTargetPosition=[0.5, 0, 0.08],
)
pb.configureDebugVisualizer(
    pb.COV_ENABLE_SHADOWS, 1, lightPosition=[50, 0, 80],
)

planeId = pb.loadURDF("plane.urdf")  # Create plane

robotURDF = "../assets/robots/sawyer_openhand.urdf"

# robotID = pb.loadURDF(robotURDF, useFixedBase=True)
robot = px.Body(robotURDF, use_fixed_base=True)
# Add object to pybullet and tacto simulator
urdfObj = "../assets/objects/sphere_small.urdf"

objStartPos = [0.50, 0, 0.07]
objStartOrientation = pb.getQuaternionFromEuler([0, 0, np.pi / 2])

obj = px.Body(urdf_path=urdfObj,
              base_position=objStartPos,
              base_orientation=objStartOrientation,
              global_scaling=1.0,
              use_fixed_base=False)

objID = obj.id
allsights.add_body(obj)


robotID = robot.id
rob = Robot(robotID, objID)

allsight_joints = ["finger_3_2_to_finger_3_3", "finger_2_2_to_finger_2_3", "finger_1_2_to_finger_1_3"]
sensorID = rob.get_id_by_name(allsight_joints)
sensorLinks = rob.get_id_by_name(allsight_joints)

allsights.add_camera(robotID, sensorLinks)

cam = Camera()
color, depth = cam.get_image()

rob.go(rob.pos, wait=False)

nbJoint = pb.getNumJoints(robotID)

time_render = []
time_vis = []

dz = 0.003
interval = 10
posList = [
    [0.50, 0, 0.205],
    [0.50, 0, 0.213],
    [0.50, 0.03, 0.205],
    [0.50, 0.03, 0.213],
]
posID = 0
pos = posList[posID].copy()

t = px.utils.SimulationThread(real_time_factor=1.0)
t.start()

# t=0
gripForce = 20
for i in range(5):
    color, depth = allsights.render()
    allsights.updateGUI(color, depth)
    time.sleep(0.05)

normalForceList0 = []
normalForceList1 = []

print("\n")

while True:


    rob.gripper_open()
    rob.go(pos, wait=True)
    rob.grasp()
    # Record sensor states
    tactileColor, tactileDepth = allsights.render()
    tactileColorL, tactileColorR = tactileColor[0], tactileColor[1]
    tactileDepthL, tactileDepthR = tactileDepth[0], tactileDepth[1]

    visionColor, visionDepth = cam.get_image()
    allsights.updateGUI(tactileColor, tactileDepth)

    normalForce0, lateralForce0 = get_forces(robotID, objID, sensorID[0], -1)
    normalForce1, lateralForce1 = get_forces(robotID, objID, sensorID[1], -1)
    normalForce = [normalForce0, normalForce1]
    # print("Normal Force Left", normalForce0, "Normal Force Right", normalForce1)
    # print("normal force", normalForce, "lateral force", lateralForce)

    objPos0, objOri0 = get_object_pose()

    pos[-1] += dz

    rob.go(pos, wait=True)

    # Save the data
    objPos, objOri = get_object_pose()

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
        visionColor,
        visionDepth,
        gripForce,
        normalForce,
        label,
    )
    print("\rsample {}".format(log.id * log.batch_size + len(log.dataList)), end="")

    # Reset
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

    # pb.resetBasePositionAndOrientation(objID, objRestartPos, objRestartOrientation)
    obj.set_base_pose(objRestartPos, objRestartOrientation)

    for i in range(10):
        pb.stepSimulation()

    st = time.time()
    time_render.append(time.time() - st)
    time_render = time_render[-100:]
    # print("render {:.4f}s".format(np.mean(time_render)), end=" ")
    st = time.time()


    time_vis.append(time.time() - st)
    time_vis = time_vis[-100:]

    # print("visualize {:.4f}s".format(np.mean(time_vis)))

    color, depth = allsights.render()
    allsights.updateGUI(color, depth)
    time.sleep(0.05)

pb.disconnect()  # Close PyBullet
