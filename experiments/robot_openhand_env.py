# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pybullet as pb
# from pyquaternion import Quaternion
from scipy.spatial import ConvexHull, distance
from time import time
from time import sleep
import logging
import os

# import deepdish as dd
import numpy as np
import pybullet as pb
import pybullet_data
import pybulletX as px
from tacto_allsight_wrapper import allsight_wrapper
import sys

PATH = os.path.join(os.path.dirname(__file__), "../")
sys.path.insert(0, PATH)

import sys
import cv2


class InsertionEnv:

    def __init__(self, robot_name='panda',
                 obj_start_pos=None,
                 obj_start_ori=None,
                 allsight_display=False):

        # Init finger tips
        rel_path = os.path.join(os.path.dirname(__file__), "../")
        sys.path.insert(0, rel_path)
        bg = cv2.imread(os.path.join(rel_path, f"experiments/conf/ref/ref_frame_rrrgggbbb.jpg"))
        conf_path = os.path.join(rel_path, f"experiments/conf/sensor/config_allsight_rrrgggbbb.yml")

        self.allsights = allsight_wrapper.Sensor(
            width=224, height=224, visualize_gui=allsight_display, **{"config_path": conf_path},
            background=bg if True else None
        )

        # Init PyBullet
        cid = px.init()
        print(pb.getPhysicsEngineParameters(cid))

        # logging.info("Initializing world")
        # physicsClient = pb.connect(pb.DIRECT)
        pb.setPhysicsEngineParameter(solverResidualThreshold=0)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        # pb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth

        pb.resetDebugVisualizerCamera(
            cameraDistance=0.6,
            cameraYaw=15,
            cameraPitch=-20,
            cameraTargetPosition=[0.5, 0, 0.08],
        )

        pb.configureDebugVisualizer(
            pb.COV_ENABLE_SHADOWS, 1, lightPosition=[50, 0, 80],
        )

        # Define Robot
        if robot_name == 'sawyer':
            robotURDF = "../assets/robots/sawyer_openhand.urdf"
            self.armNames = ["right_j" + str(i) for i in range(7)]
            self.armHome = [-0.01863332, -1.30851021, -0.55159919, 1.58025131, 0.14144625, 1.33963365, -1.98302146, ]
            ee_name = 'right_hand'
        elif robot_name == 'panda':
            robotURDF = "../assets/robots/panda_openhand.urdf"
            self.armNames = ["panda_joint" + str(i) for i in range(1, 8)]
            ee_name = 'panda_grasptarget_hand'
        elif robot_name == 'kuka':
            robotURDF = "../assets/robots/iiwa14_openhand.urdf"
            ee_name = 'iiwa_gripper_tip_joint'
            self.armNames = ["iiwa_joint_" + str(i) for i in range(1, 8)]
            self.armHome = [0, -0.5592432, 0, 1.733180, 0, -0.8501557, 0]

        self.robot = px.Body(robotURDF, use_fixed_base=True, flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

        # Define scene
        urdfObj = "../assets/objects/rect_small.urdf"
        if obj_start_ori is None:
            obj_start_ori = [0, 0, np.pi / 2]
        if obj_start_pos is None:
            obj_start_pos = [0.50, 0, 0.07]

        self.obj = px.Body(urdf_path=urdfObj,
                           base_position=obj_start_pos,
                           base_orientation=pb.getQuaternionFromEuler(obj_start_ori),
                           global_scaling=1.0,
                           use_fixed_base=False,
                           flags=pb.URDF_USE_INERTIA_FROM_FILE)

        self.objID = self.obj.id
        self.robotID = self.robot.id
        self.armJoints = self.get_id_by_name(self.armNames)
        self.armControlID = self.get_control_id_by_name(self.armNames)

        # Get link/joint ID for gripper
        gripperNames = [
            "finger_1_1_to_finger_1_2",
            "finger_1_2_to_finger_1_3",
            "finger_2_1_to_finger_2_2",
            "finger_2_2_to_finger_2_3",
            "base_to_finger_3_2",
            "finger_3_2_to_finger_3_3"
        ]
        gripperBases = ['base_to_finger_1_1',
                        'base_to_finger_2_1']

        allsight_joints = ["finger_3_2_to_finger_3_3",
                           "finger_2_2_to_finger_2_3",
                           "finger_1_2_to_finger_1_3"]

        self.gripperJoints = self.get_id_by_name(gripperNames)
        self.gripperControlID = self.get_control_id_by_name(gripperNames)

        self.gripperBaseJoints = self.get_id_by_name(gripperBases)
        self.gripperBaseControlID = self.get_control_id_by_name(gripperBases)

        # Get ID for end effector
        self.eefID = self.get_id_by_name([ee_name])[0]

        self.pos = [0.53, 0, 0.215]
        self.ori = [0, 3.14, np.pi / 2]
        self.tol = 1e-9

        # add object to allsight
        self.allsights.add_body(self.obj)
        self.sensorLinks = self.get_id_by_name(allsight_joints)
        self.allsights.add_camera(self.robotID, self.sensorLinks)

        ###########################
        # Insertion specific

        ###########################

        self.init_robot()

    def get_id_by_name(self, names):
        """
        get joint/link ID by name
        """
        nbJoint = pb.getNumJoints(self.robotID)
        jointNames = {}
        for i in range(nbJoint):
            name = pb.getJointInfo(self.robotID, i)[1].decode()
            jointNames[name] = i

        return [jointNames[name] for name in names]

    def get_control_id_by_name(self, names):
        """
        get joint/link ID by name
        """
        nbJoint = pb.getNumJoints(self.robotID)
        jointNames = {}
        ctlID = 0
        for i in range(nbJoint):
            jointInfo = pb.getJointInfo(self.robotID, i)

            name = jointInfo[1].decode("utf-8")

            # skip fixed joint
            if jointInfo[2] == 4:
                continue

            # skip base joint
            # if jointInfo[-1] == -1:
            #     continue

            jointNames[name] = ctlID
            ctlID += 1

        return [jointNames[name] for name in names]

    def init_robot(self):
        self.reset_robot()

        self.xin = pb.addUserDebugParameter("x", 0.3, 0.85, self.pos[0])
        self.yin = pb.addUserDebugParameter("y", -0.85, 0.85, self.pos[1])
        self.zin = pb.addUserDebugParameter("z", 0.0, 0.8, self.pos[2])

        self.go(self.pos, self.ori)

    def reset_robot(self):
        for j in range(len(self.armJoints)):
            pb.resetJointState(self.robotID, self.armJoints[j], self.armHome[j])

        pb.resetJointState(self.robotID, self.gripperBaseJoints[0], 0.7)
        pb.resetJointState(self.robotID, self.gripperBaseJoints[1], -0.7)

        self.gripper_open()

        for _ in range(5):
            color, depth = self.allsights.render()
            self.allsights.updateGUI(color, depth)

    # Get the position and orientation of the UR5 end-effector
    def get_ee_pose(self):
        res = pb.getLinkState(self.robotID, self.eefID)

        world_positions = res[0]
        world_orientations = res[1]

        return (world_positions, world_orientations)

    # Get the joint angles (6 ur5 joints)
    def get_joint_angles(self):
        joint_angles = [_[0] for _ in pb.getJointStates(self.robotID, self.armJoints)]
        return joint_angles

    # Go to the target pose
    def go(self, pos, ori=None, wait=False, gripForce=20):

        if ori is None:
            ori = self.ori
        ori_q = pb.getQuaternionFromEuler(ori)

        jointPose = pb.calculateInverseKinematics(self.robotID, self.eefID, pos, ori_q)
        jointPose = np.array(jointPose)

        self.cur_joint_pose = jointPose.copy()

        maxForces = np.ones(len(jointPose)) * 200

        # Select the relevant joints for arm
        jointPose = jointPose[self.armControlID]
        maxForces = maxForces[self.armControlID]

        pb.setJointMotorControlArray(
            self.robotID,
            tuple(self.armJoints),
            pb.POSITION_CONTROL,
            targetPositions=jointPose,
            forces=maxForces,
            targetVelocities=np.zeros((7,)),
            # positionGains=np.ones((7,)) * 0.03,
            # velocityGains=np.ones((7,)),
        )

        self.pos = pos
        if ori is not None:
            self.ori = ori

        if wait:
            last_err = 1e6
            while True:
                pb.stepSimulation()
                ee_pose = self.get_ee_pose()
                err = (
                        np.sum(np.abs(np.array(ee_pose[0]) - pos))
                        + np.sum(np.abs(np.array(ee_pose[1]) - ori_q))
                )
                diff_err = last_err - err
                last_err = err

                if np.abs(diff_err) < self.tol:
                    break

    # Go to the target pose
    def grasp(self, gripForce=100):

        finish_time = time() + 5.0
        has_contact = 0

        while time() < finish_time and (has_contact < 3):
            pb.stepSimulation()
            # joint_info_list = [pb.getJointState(self.robotID, joint_index).joint_position + 0.001 for joint_index in
            #                    self.gripperJoints]

            for joint in self.gripperJoints:
                pb.setJointMotorControl2(bodyUniqueId=self.robotID,
                                         jointIndex=joint,
                                         controlMode=pb.VELOCITY_CONTROL,
                                         targetVelocity=0.2,
                                         force=gripForce)

                # pb.setJointMotorControl2(targetPosition=pb.getJointState(self.robotID, joint).joint_position + 0.001,
                #                         bodyIndex=self.robotID,
                #                         jointIndex=joint,
                #                         controlMode=pb.POSITION_CONTROL, maxVelocity=1, force=10)

                # pb.setJointMotorControl2(bodyIndex=self.robotID, jointIndex=joint,
                #                         controlMode=pb.POSITION_CONTROL,
                #                         targetPosition=pb.getJointState(self.robotID, joint).joint_position + 0.001,
                #                         # positionGain=0.5, velocityGain=0.5, force=1.5,
                #                         # force=1,
                #                         force=gripForce)

                # pb.setJointMotorControl2(bodyIndex=self.robotID,
                #                           jointIndex=joint,
                #                           controlMode=pb.POSITION_CONTROL,
                #                           targetPosition=pb.getJointState(self.robotID, joint).joint_position + 0.03,
                #                           targetVelocity=0,
                #                           force=50,
                #                           positionGain= 0.03,
                #                           velocityGain=1)

            has_contact = len(pb.getContactPoints(self.robotID, self.objID))

        for joint in self.gripperJoints:
            pb.setJointMotorControl2(bodyUniqueId=self.robotID,
                                     jointIndex=joint,
                                     controlMode=pb.POSITION_CONTROL,
                                     targetVelocity=0.2,
                                     targetPosition=pb.getJointState(self.robotID, joint).joint_position,
                                     force=gripForce)

        # color, depth = self.allsights.render()
        # self.allsights.updateGUI(color, depth)

    # Gripper open
    def gripper_open(self):

        pb.setJointMotorControlArray(
            self.robotID,
            self.gripperJoints,
            pb.POSITION_CONTROL,
            targetPositions=np.zeros(len(self.gripperJoints)).tolist(),
        )

    def get_object_pose(self):
        res = pb.getBasePositionAndOrientation(self.objID)
        world_positions = res[0]
        world_orientations = res[1]
        world_positions = np.array(world_positions)
        world_orientations = np.array(world_orientations)

        return (world_positions, world_orientations)
