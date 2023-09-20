# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pybullet as pb
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull, distance
from time import time

force_pyramid_sides = 9
force_pyramid_radius = .06


class Robot:
    def __init__(self, robotID, objID):
        self.robotID = robotID

        # Get link/joint ID for arm
        self.armNames = [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]
        self.armJoints = self.get_id_by_name(self.armNames)
        self.armControlID = self.get_control_id_by_name(self.armNames)

        # Get link/joint ID for gripper
        self.gripperNames = [
            "finger_2_2_to_finger_2_3",
            "finger_3_2_to_finger_3_3",
        ]

        self.gripperNames = [
            "base_to_finger_1_1",
            "finger_1_1_to_finger_1_2",
            "finger_1_2_to_finger_1_3",
            "base_to_finger_2_1",
            "finger_2_1_to_finger_2_2",
            "finger_2_2_to_finger_2_3",
            "base_to_finger_3_2",
            "finger_3_2_to_finger_3_3"
        ]

        self.gripperJoints = self.get_id_by_name(self.gripperNames)
        self.gripperControlID = self.get_control_id_by_name(self.gripperNames)

        # Get ID for end effector
        self.eeName = ["right_hand"]
        self.eefID = self.get_id_by_name(self.eeName)[0]

        self.armHome = [
            -0.01863332,
            -1.30851021,
            -0.55159919,
            1.58025131,
            0.14144625,
            1.33963365,
            -1.98302146,
        ]
        self.pos = [0.53, 0, 0.215]
        self.ori = [0, 3.14, np.pi / 2]
        self.width = 0.11

        self.tol = 1e-9
        self.objID = objID
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
            if jointInfo[-1] == -1:
                continue

            jointNames[name] = ctlID
            ctlID += 1

        return [jointNames[name] for name in names]

    def reset_robot(self):
        for j in range(len(self.armJoints)):
            pb.resetJointState(self.robotID, self.armJoints[j], self.armHome[j])

    def init_robot(self):
        self.reset_robot()

        self.xin = pb.addUserDebugParameter("x", 0.3, 0.85, self.pos[0])
        self.yin = pb.addUserDebugParameter("y", -0.85, 0.85, self.pos[1])
        self.zin = pb.addUserDebugParameter("z", 0.0, 0.8, self.pos[2])
        self.widthin = pb.addUserDebugParameter("width", 0.03, 0.11, self.width)

        self.go(self.pos, self.ori, self.width)

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

    # Get the gripper width gripper width
    def get_gripper_width(self):
        width = 2 * np.abs(pb.getJointState(self.robotID, self.gripperJoints[-1])[0])
        return width

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

        jointPose = self.cur_joint_pose
        jointPose[self.gripperControlID] = 1.0

        maxForces = np.zeros(len(jointPose))
        maxForces[self.gripperControlID] = gripForce

        # Select the relevant joints for arm and gripper
        jointPose = jointPose[self.gripperControlID]
        maxForces = maxForces[self.gripperControlID]

        # pb.setJointMotorControlArray(
        #     self.robotID,
        #     tuple(self.gripperJoints),
        #     pb.POSITION_CONTROL,
        #     targetPositions=jointPose,
        #     forces=maxForces,
        # )

        finish_time = time() + 10.0
        has_contact = 0
        while time() < finish_time and (has_contact < 2):
            pb.stepSimulation()
            for joint in self.gripperJoints:
                pb.setJointMotorControl2(bodyUniqueId=self.robotID,
                                         jointIndex=joint,
                                         controlMode=pb.VELOCITY_CONTROL,
                                         targetVelocity=0.05,
                                         force=gripForce)
            has_contact = len(pb.getContactPoints(self.robotID, self.objID))

    def update_input(self):

        x = pb.readUserDebugParameter(self.xin)
        y = pb.readUserDebugParameter(self.yin)
        z = pb.readUserDebugParameter(self.zin)

        pos = [x, y, z]
        self.go(pos, wait=False)

    # Gripper close
    def gripper_close(self, width=None, max_torque=None):
        pass

    # Gripper open
    def gripper_open(self):
        pb.setJointMotorControlArray(
            self.robotID,
            self.gripperJoints,
            pb.POSITION_CONTROL,
            targetPositions=np.zeros(len(self.gripperJoints)).tolist(),
        )

#     def grip_qual(self):
#         """
#         evaluate the grasp quality
#         """
#         contact = pb.getContactPoints(self.objID,
#                                       self.robotID)  # see if hand is still holding obj after gravity is applied
#         if len(contact) > 0:
#             force_torque = self.gws_pyramid_extension()
#             # print("force_torque: ", force_torque)
#             # print("force_torque shape: ", np.array(force_torque).shape)
#             vol = self.volume(force_torque)
#             # print("volume: ", vol)
#             ep = self.eplison(force_torque)
#             # print("eplison: ", ep)
#         else:
#             vol = None
#             ep = None
#         return vol, ep
#
#     def volume(self, force_torque):
#         """
#         get qhull of the 6 dim vectors [fx, fy, fz, tx, ty, tz] created by gws (from contact points)
#         get the volume
#         """
#         vol = ConvexHull(points=force_torque)
#         return vol.volume
#
#     def eplison(self, force_torque):
#         """
#         get qhull of the 6 dim vectors [fx, fy, fz, tx, ty, tz] created by gws (from contact points)
#         get the distance from centroid of the hull to the closest vertex
#         """
#         hull = ConvexHull(points=force_torque)
#         centroid = []
#         for dim in range(0, 6):
#             centroid.append(np.mean(hull.points[hull.vertices, dim]))
#         shortest_distance = 500000000
#         closest_point = None
#         for point in force_torque:
#             point_dist = distance.euclidean(centroid, point)
#             if point_dist < shortest_distance:
#                 shortest_distance = point_dist
#                 closest_point = point
#
#         return shortest_distance
#
#     def get_obj_info(self):
#         """
#         get object data to figure out how far away the hand needs to be to make its approach
#         """
#         obj_data = pb.getCollisionShapeData(self.objID, -1)[0]
#         geometry_type = obj_data[2]
#         # print("geometry type: " + str(geometry_type))
#         dimensions = obj_data[3]
#         # print("dimensions: "+ str(dimensions))
#         local_frame_pos = obj_data[5]
#         # print("local frome position: " + str(local_frame_pos))
#         local_frame_orn = obj_data[6]
#         # print("local frame oren: " + str(local_frame_orn))
#         import math
#         diagonal = math.sqrt(dimensions[0] ** 2 + dimensions[1] ** 2 + dimensions[2] ** 2)
#         # print("diagonal: ", diagonal)
#         max_radius = diagonal / 2
#         return local_frame_pos, max_radius
#
#     def gws_pyramid_extension(self):
#
#         # often dont have enough contact points to create a qhull of the right dimensions,
#         # so create more that are very close to the existing ones
#         local_frame_pos, max_radius = self.get_obj_info()
#         # sim uses center of mass as a reference for the Cartesian world transforms in getBasePositionAndOrientation
#         obj_pos, obj_orn = pb.getBasePositionAndOrientation(self.objID)
#         force_torque = []
#         contact_points = pb.getContactPoints(self.robotID, self.objID)
#         for point in contact_points:
#             contact_pos = point[6]
#             normal_vector_on_obj = point[7]
#             normal_force_on_obj = point[9]
#             force_vector = np.array(normal_vector_on_obj) * normal_force_on_obj
#             if np.linalg.norm(force_vector) > 0:
#                 new_vectors = self.get_new_normals(force_vector, normal_force_on_obj, force_pyramid_sides,
#                                                    force_pyramid_radius)
#
#                 radius_to_contact = np.array(contact_pos) - np.array(obj_pos)
#
#                 for pyramid_vector in new_vectors:
#                     torque_numerator = np.cross(radius_to_contact, pyramid_vector)
#                     torque_vector = torque_numerator / max_radius
#                     force_torque.append(np.concatenate([pyramid_vector, torque_vector]))
#
#         return force_torque
#
#     def get_new_normals(self, force_vector, normal_force, sides, radius):
#         """
#         utility function to help with GWS/pyramid extension for contact points
#         """
#         return_vectors = []
#         # get arbitrary vector to get cross product which should be orthogonal to both
#         vector_to_cross = np.array((force_vector[0] + 1, force_vector[1] + 2, force_vector[2] + 3))
#         orthg = np.cross(force_vector, vector_to_cross)
#         orthg_vector = (orthg / np.linalg.norm(orthg)) * radius
#         rot_angle = (2 * np.pi) / sides
#         split_force = normal_force / sides
#
#         for side_num in range(0, sides):
#             rotated_orthg = Quaternion(axis=force_vector, angle=(rot_angle * side_num)).rotate(orthg_vector)
#             new_vect = force_vector + np.array(rotated_orthg)
#             norm_vect = (new_vect / np.linalg.norm(new_vect)) * split_force
#             return_vectors.append(norm_vect)
#
#         return return_vectors
#
#     def relax(self):
#         """
#         return all joints to neutral/furthest extended, based on urdf specification
#         """
#         joint = 0
#         num = pb.getNumJoints(self.robotID)
#         while joint < num:
#             pb.resetJointState(self.robotID, jointIndex=joint, targetValue=0.0)
#             joint = joint + 1
