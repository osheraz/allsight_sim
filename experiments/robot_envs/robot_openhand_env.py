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
        bg = cv2.imread(os.path.join(rel_path, f"conf/ref/ref_frame_white15.jpg"))
        conf_path = os.path.join(rel_path, f"conf/sensor/config_allsight_white.yml")

        self.allsights = allsight_wrapper.Sensor(
            width=224, height=224,
            visualize_gui=allsight_display, **{"config_path": conf_path},
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
            self.armHome = [-0.0186, -1.308, -0.5515, 1.580, 0.1414, 1.3396, -1.983]
            ee_name = 'right_hand'
        elif robot_name == 'panda':
            robotURDF = "../assets/robots/panda_openhand.urdf"
            self.armNames = ["panda_joint" + str(i) for i in range(1, 8)]
            ee_name = 'panda_grasptarget_hand'
        elif robot_name == 'kuka':
            robotURDF = "../assets/robots/iiwa14_openhand.urdf"
            ee_name = 'iiwa_gripper_tip_joint'
            self.armNames = ["iiwa_joint_" + str(i) for i in range(1, 8)]
            # self.armHome = [1.080, 0.7178, 1.7055, 2.068, 0.8235, -1.10589, 0.8079]
            self.armHome = [1.082395318127047, 0.7183057710476938, 1.6956706531485544, 2.0613745662841523, 0.8197305416268088, -1.1176214910072408, 0.8086760317826353]

        self.robot = px.Body(robotURDF, use_fixed_base=True, flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

        # Define scene
        urdfObj = "../assets/objects/rect_grasp.urdf"
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

        self.proximal_joints = ['finger_1_1_to_finger_1_2', 'base_to_finger_3_2', 'finger_2_1_to_finger_2_2']
        self.distal_joints = ["finger_1_2_to_finger_1_3", "finger_2_2_to_finger_2_3", "finger_3_2_to_finger_3_3"]

        gripper_bases = ['base_to_finger_1_1', 'base_to_finger_2_1']

        allsight_joints = ["finger_3_2_to_finger_3_3", "finger_2_2_to_finger_2_3", "finger_1_2_to_finger_1_3"]

        self.proximalJointsID = self.get_id_by_name(self.proximal_joints)
        self.distalJointsID = self.get_id_by_name(self.distal_joints)
        self.gripperJointsID = self.get_id_by_name(self.proximal_joints + self.distal_joints)
        # self.gripperControlID = self.get_control_id_by_name(self.proximal_joints + self.distal_joints)

        self.gripperBaseJointsID = self.get_id_by_name(gripper_bases)
        # self.gripperBaseControlID = self.get_control_id_by_name(gripper_bases)

        # Get ID for end effector
        self.eefID = self.get_id_by_name([ee_name])[0]

        self.cur_pose = [0.53, 0, 0.215]
        self.cur_ori = [0, 3.14, np.pi / 2]
        self.tol = 1e-9

        # add object to tacto sim
        self.allsights.add_body(self.obj)
        self.sensorLinks = self.get_id_by_name(allsight_joints)
        self.allsights.add_camera(self.robotID, self.sensorLinks)

        self.reset_robot()

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

    def reset_robot(self):

        # init robot arm
        for j in range(len(self.armJoints)):
            pb.resetJointState(self.robotID, self.armJoints[j], self.armHome[j])

        # init finger rotation angle
        pb.resetJointState(self.robotID, self.gripperBaseJointsID[0], 0.7)
        pb.resetJointState(self.robotID, self.gripperBaseJointsID[1], -0.7)

        # init the rest of the gripper joints
        self.gripper_open()

        # update tactile images
        for _ in range(5):
            color, depth = self.allsights.render()
            self.allsights.updateGUI(color, depth)

    # Get the position and orientation of the robot end-effector
    def get_ee_pose(self):
        res = pb.getLinkState(self.robotID, self.eefID)

        world_positions = res[0]
        world_orientations = res[1]

        return (world_positions, world_orientations)

    # Get the joint angles
    def get_joint_angles(self):
        joint_angles = [_[0] for _ in pb.getJointStates(self.robotID, self.armJoints)]
        return joint_angles

    def get_gripper_angles(self):
        joint_angles = [_[0] for _ in pb.getJointStates(self.robotID, self.gripperJointsID)]
        return joint_angles

    # Gripper open
    def gripper_open(self, wait=True):

        # pb.setJointMotorControlArray(
        #     self.robotID,
        #     self.gripperJointsID,
        #     pb.POSITION_CONTROL,
        #     targetPositions=np.zeros(len(self.gripperJointsID)).tolist(),
        # )

        # for i in range(len(self.gripperJointsID)):
        #     pb.setJointMotorControl2(targetPosition=0.,
        #                              bodyIndex=self.robotID,
        #                              jointIndex=self.gripperJointsID[i],
        #                              controlMode=pb.POSITION_CONTROL,
        #                              maxVelocity=0.25,
        #                              force=10)

        for i in range(len(self.proximalJointsID)):
            pb.setJointMotorControl2(targetPosition=0,
                                     bodyIndex=self.robotID,
                                     jointIndex=self.proximalJointsID[i],
                                     controlMode=pb.POSITION_CONTROL,
                                     maxVelocity=0.25,
                                     force=5)

        for i in range(len(self.distalJointsID)):
            pb.setJointMotorControl2(targetPosition=0.,
                                     bodyIndex=self.robotID,
                                     jointIndex=self.distalJointsID[i],
                                     controlMode=pb.POSITION_CONTROL,
                                     maxVelocity=0.25,
                                     force=2)


        if wait:
            last_err = 1e6
            while True:
                pb.stepSimulation()
                gripper_angles = self.get_gripper_angles()
                err = np.linalg.norm(np.array(gripper_angles))
                diff_err = last_err - err
                last_err = err
                if np.abs(diff_err) < self.tol:
                    break

    def get_object_pose(self):
        res = pb.getBasePositionAndOrientation(self.objID)
        world_positions = res[0]
        world_orientations = res[1]
        world_positions = np.array(world_positions)
        world_orientations = np.array(world_orientations)

        return (world_positions, world_orientations)

    # Go to the target pose
    def go(self, pos, ori=None, ori_is_quat=False, wait=False):

        # keep the same ori if not specified
        if ori is None:
            ori = self.cur_ori

        if ori_is_quat:
            ori_q = ori
            ori = pb.getEulerFromQuaternion(ori)
        else:
            ori_q = pb.getQuaternionFromEuler(ori)

        joint_pose = pb.calculateInverseKinematics(self.robotID,
                                                   self.eefID,
                                                   pos,
                                                   ori_q,
                                                   residualThreshold=0.001,
                                                   maxNumIterations=100)

        joint_pose = np.array(joint_pose)
        max_force = np.ones(len(joint_pose)) * 200

        # Select the relevant joints for arm
        joint_pose = joint_pose[self.armControlID]
        max_force = max_force[self.armControlID]

        # Move the arm to the desired angle using position control
        # pb.setJointMotorControlArray(
        #     self.robotID,
        #     tuple(self.armJoints),
        #     pb.POSITION_CONTROL,
        #     targetPositions=joint_pose,
        #     forces=max_force,
        #     targetVelocities=np.zeros((7,)),
        #     positionGains=np.ones((7,)) * 0.03,
        #     velocityGains=np.ones((7,)),
        # )
        self.MAX_VELOCITY = 0.25
        self.MAX_FORCE = 150

        for i in range(len(joint_pose)):
            pb.setJointMotorControl2(targetPosition=joint_pose[i],
                                     bodyIndex=self.robotID,
                                     jointIndex=self.armJoints[i],
                                     controlMode=pb.POSITION_CONTROL,
                                     maxVelocity=self.MAX_VELOCITY,
                                     force=self.MAX_FORCE)

        # update current pos and orientation
        self.cur_pose = pos
        if ori is not None:
            self.cur_ori = ori

        # make sure we get there
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

    def grasp(self, gripForce=7):

        max_grasp_time = time() + 20.0
        has_contact = 0

        while time() < max_grasp_time and (has_contact < 3):
            pb.stepSimulation()
            # joint_info_list = [pb.getJointState(self.robotID, joint_index).joint_position + 0.001 for joint_index in
            #                    self.gripperJointsID]

            for joint in self.proximalJointsID:
                vel = 0.2
                pb.setJointMotorControl2(bodyUniqueId=self.robotID,
                                         jointIndex=joint,
                                         controlMode=pb.VELOCITY_CONTROL,
                                         targetVelocity=vel,
                                         force=gripForce)

            for joint in self.distalJointsID:
                vel = 0.05
                pb.setJointMotorControl2(bodyUniqueId=self.robotID,
                                         jointIndex=joint,
                                         controlMode=pb.VELOCITY_CONTROL,
                                         targetVelocity=vel,
                                         force=gripForce)

            has_contact = len(pb.getContactPoints(self.robotID, self.objID))

            sleep(0.02)
            color, depth = self.allsights.render()
            self.allsights.updateGUI(color, depth)

        for joint in self.gripperJointsID:
            pb.setJointMotorControl2(bodyUniqueId=self.robotID,
                                     jointIndex=joint,
                                     controlMode=pb.POSITION_CONTROL,
                                     targetVelocity=0.2,
                                     targetPosition=pb.getJointState(self.robotID, joint).joint_position,
                                     force=gripForce)

        color, depth = self.allsights.render()
        self.allsights.updateGUI(color, depth)

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
