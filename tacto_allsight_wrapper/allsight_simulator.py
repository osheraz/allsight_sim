# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
import time
import pybullet as pyb

import cv2
import pybulletX as px
import os
import numpy as np
from typing import Any
from omegaconf import DictConfig
import shutup;

shutup.please()

log = logging.getLogger(__name__)

from tacto_allsight_wrapper import allsight_wrapper

# import allsight wrapper
PATH = os.path.join(os.path.dirname(__file__), "../")
sys.path.insert(0, PATH)
from experiments.utils.logger import DataSimLogger
from experiments.utils.geometry import rotation_matrix, concatenate_matrices
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)
origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)


class Simulator:

    def __init__(self, cfg: DictConfig,
                 summary: dict,
                 with_bg=False,
                 attributes: dict = None
                 ):

        '''Simulator object for defining simulation scene with allsight sensor

        Parameters
        ----------
        cfg : DictConfig
            allsight config dict -> allsight.yaml
        with_bg : bool, optional
            Use background img, by default False
        attributes : dict, optional
            Set object attributes if needed, by default None
        '''

        # bg image
        leds = summary['leds']

        bg = cv2.imread(f"conf/ref_frame_{leds}.jpg")
        conf_path = f"conf/config_allsight_{leds}.yml"

        # initialize allsight
        # TODO: modify to support different leds
        self.allsight = allsight_wrapper.Sensor(
            **cfg.tacto, **{"config_path": conf_path},
            background=bg if with_bg else None
        )

        self.summary = summary
        self.HH = 20
        self.start_h = 0.012
        #                    x  y  z              h      r
        self.finger_props = [0, 0, self.start_h, 0.016, 0.0128]  # [m]

    # visual creator function
    def create_env(self, cfg: DictConfig, obj_id: str = '30'):
        """Create scene including visualizer gui and bodys

        Parameters
        ----------
        cfg : DictConfig
            allsight config dict -> allsight.yaml
        obj_id : str, optional
            Define which size of sphere to use, by default '30'
        """

        # Initialize World
        log.info("Initializing world")
        px.init()
        pyb.resetDebugVisualizerCamera(**cfg.pybullet_camera)
        pyb.configureDebugVisualizer(
            pyb.COV_ENABLE_SHADOWS, 1, lightPosition=[50, 0, 80],
        )

        self.body = px.Body(**cfg.allsight)

        # object body 
        # take the urdf path with the relevant id given
        # obj_urdf_path = f"../assets/objects/sphere_{obj_id}.urdf"
        obj_urdf_path = f"../assets/objects/cube_small.urdf"

        cfg.object.urdf_path = obj_urdf_path
        self.obj = px.Body(**cfg.object)
        # set start pose
        self.obj.set_base_pose([0, 0, 0.056])
        self.allsight.add_body(self.obj)

        # camera body
        self.allsight.add_camera(self.body.id, [-1])

        # Create control panel to control the 6DoF pose of the object
        self.panel = px.gui.PoseControlPanel(self.obj, **cfg.object_control_panel)
        self.panel.start()

    def start(self):
        '''Start the simulation thread
        '''

        self.t = px.utils.SimulationThread(real_time_factor=1.0)
        self.t.start()

    def run_sim(self):
        '''run simulation loop for demo proposes
        '''

        self.start()
        while True:
            color, depth = self.allsight.render()
            self.allsight.updateGUI(color, depth)

        self.t.stop()

    def collect_data(self, conf):

        # start the simulation thread 
        self.start()

        # create constraints
        (self._obj_x, self._obj_y, self._obj_z), self._obj_or = self.obj.get_base_pose()
        init_xyz = [self._obj_x, self._obj_y, self._obj_z]

        self.cid = pyb.createConstraint(
            self.obj.id,  # parent body unique id
            -1,  # parent link index (or -1 for the base)
            -1,  # child body unique id, or -1 for no body (specify anon-dynamic child frame in world coordinates)
            -1,  # child link index, or -1 for the base
            pyb.JOINT_FIXED,  # joint type: JOINT_PRISMATIC, JOINT_FIXED,JOINT_POINT2POINT, JOINT_GEAR
            [0, 0, 0],  # joint axis, in child link frame
            [0, 0, 0],  # position of the joint frame relative to parent center of mass frame.
            init_xyz
            # position of the joint frame relative to a given child center of mass frame (or world origin if no child specified)
        )

        # create data logger object
        self.logger = DataSimLogger(conf['leds'], conf['indenter'], save=conf['save'], save_depth=False)

        # take ref frame 
        ref_frame, _ = self.allsight.render()

        ref_img_color_path = os.path.join(self.logger.dataset_path_images, 'ref_frame.jpg')

        if conf['save']:
            ref_frame[0] = cv2.cvtColor(ref_frame[0], cv2.COLOR_BGR2RGB)
            if not cv2.imwrite(ref_img_color_path, ref_frame[0]):
                raise Exception("Could not write image")

        frame_count = 0

        Q = np.linspace(0, 2 * np.pi, conf['N'])

        for q in Q:

            for i in range(conf['start_from'], self.HH + 5 - 1, 1):

                if i == self.HH: continue

                f_push = 80 if i < self.HH else 70

                push_point_start, push_point_end = self.get_push_point_by_index(q, i)

                # pyb.changeConstraint(self.cid, jointChildPivot=push_point_start[0],
                #                      jointChildFrameOrientation=push_point_start[1],
                #                      maxForce=f_push)

                pyb.createConstraint(self.obj.id, -1, -1, -1, pyb.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                     childFramePosition=push_point_start[0], childFrameOrientation=push_point_start[1])

                color, depth = self.allsight.render()
                self.allsight.updateGUI(color, depth)
                time.sleep(0.01)

                # pyb.changeConstraint(self.cid, jointChildPivot=push_point_end[0],
                #                      jointChildFrameOrientation=push_point_end[1],
                #                      maxForce=f_push)

                pyb.createConstraint(self.obj.id, -1, -1, -1, pyb.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                     childFramePosition=push_point_end[0], childFrameOrientation=push_point_end[1])

                color, depth = self.allsight.render()
                self.allsight.updateGUI(color, depth)

                time.sleep(0.01)
                color, depth = self.allsight.render()
                self.allsight.updateGUI(color, depth)

                pose = list(pyb.getBasePositionAndOrientation(self.obj.id)[0][:3])
                # pose[0] -= np.sign(pose[0]) * 0.002  # radi
                # pose[1] -= np.sign(pose[1]) * 0.002  # radi
                # pose[2] -= self.start_h + 0.006

                orient = pyb.getBasePositionAndOrientation(self.obj.id)[1][:4]
                force = self.allsight.get_force('cam0')['2_-1']

                color_img = color[0]
                depth_img = np.concatenate(list(map(self.allsight._depth_to_color, depth)), axis=1)
                self.logger.append(i, np.interp(q, [0, 2 * np.pi], [0, 1]),
                                   color_img, depth_img, pose, orient, force, frame_count)

                frame_count += 1

            if conf['save']: self.logger.save_batch_images()

        self.logger.save_data_dict()

    def get_push_point_by_index(self, q: float, i: int) -> Any:
        '''Get start and end points for every press process

        Parameters
        ----------
        q : float
            angle id
        i : int
            _description_

        Returns
        -------
        Any
            push_point_start : list
                starting point for press process [xyz]
            push_point_end : list
                end point for press process [xyz]
        '''

        # TODO: should take everything from conf

        G = 2
        H_cyl = np.linspace(0, self.finger_props[3], self.HH)

        if i < len(H_cyl):

            H = np.asarray([
                self.finger_props[0] + self.finger_props[4] * np.cos(q),
                self.finger_props[1] + self.finger_props[4] * np.sin(q),
                self.finger_props[2] + H_cyl[i],
            ])

            H2 = np.asarray([
                self.finger_props[0] + self.finger_props[4] * G * np.cos(q),
                self.finger_props[1] + self.finger_props[4] * G * np.sin(q),
                self.finger_props[2] + H_cyl[i],
            ])

            Rz = rotation_matrix(q, zaxis)
            Rt = rotation_matrix(np.pi, zaxis)
            Rx = rotation_matrix(np.pi, xaxis)

            rot = concatenate_matrices(Rz, Rt, Rx)[:3, :3]

            rot = R.from_matrix(rot[:3, :3]).as_quat()

            push_point_end = [[H[0], H[1], H[2]], rot.tolist()]
            push_point_start = [[H2[0], H2[1], H2[2]], rot.tolist()]

        else:

            phi = np.linspace(0, np.pi / 2, 5)[::-1][i - len(H_cyl)]

            fix = 0.15e-3 if phi < np.pi/2.2 else 0.5e-3

            B = np.asarray([
                self.finger_props[4] * np.sin(phi) * np.cos(q) + fix,
                self.finger_props[4] * np.sin(phi) * np.sin(q) + fix,
                self.finger_props[2] + self.finger_props[3] + self.finger_props[4] * np.cos(phi)+ fix,
            ])

            B2 = np.asarray([
                self.finger_props[4] * G * np.sin(phi) * np.cos(q) + fix,
                self.finger_props[4] * G * np.sin(phi) * np.sin(q) + fix,
                self.finger_props[2] + self.finger_props[3] + self.finger_props[4] * G * np.cos(phi) + fix,
            ])

            Ry = rotation_matrix(phi, yaxis)
            Rz = rotation_matrix(q, zaxis)
            Rt = rotation_matrix(np.pi / 2, yaxis)

            rot = concatenate_matrices(Rz, Ry, Rt)[:3, :3]

            rot = R.from_matrix(rot).as_quat()

            push_point_end = [[B[0], B[1], B[2]], rot.tolist()]
            push_point_start = [[B2[0], B2[1], B2[2]], rot.tolist()]

        return push_point_start, push_point_end

#     def create_finger_geometry(self, Nc: int = 30, Mc: int = 5, Mr: int = 5, display: bool = False):
#         '''Calculate sensor geomtry
#
#         Parameters
#         ----------
#         Nc : int, optional
#             _description_, by default 30
#         Mc : int, optional
#             _description_, by default 5
#         Mr : int, optional
#             _description_, by default 5
#         display : bool, optional
#             Display geomtry 3d plot, by default False
#         '''
#         self.start_h = 0
#         G = 2
#         # Init data dictionary
#         self.push_points_end = []
#         self.push_points_start = []
#         self.theta = np.linspace(0, 2 * np.pi, Nc)
#         self.H = np.linspace(self.start_h, self.finger_props[3] + self.start_h, Mc)
#
#         for j, h in enumerate(self.H):
#
#             for i, q in enumerate(self.theta):
#                 H = np.asarray([
#                     self.finger_props[0] + self.finger_props[4] * np.cos(q),
#                     self.finger_props[1] + self.finger_props[4] * np.sin(q),
#                     self.finger_props[2] + h,
#                 ])
#
#                 H2 = np.asarray([
#                     self.finger_props[0] + self.finger_props[4] * G * np.cos(q),
#                     self.finger_props[1] + self.finger_props[4] * G * np.sin(q),
#                     self.finger_props[2] + h,
#                 ])
#
#                 Rz = rotation_matrix(q, zaxis)
#                 Rt = rotation_matrix(np.pi, zaxis)
#                 Rx = rotation_matrix(np.pi, xaxis)
#
#                 rot = concatenate_matrices(Rz, Rt, Rx)[:3, :3]
#
#                 rot = R.from_matrix(rot[:3, :3]).as_quat()
#
#                 self.push_points_end.append([[H[0], H[1], H[2]], rot.tolist()])
#                 self.push_points_start.append([[H2[0], H2[1], H2[2]], rot.tolist()])
#
#         # Sphere push points
#         phi = np.linspace(0, np.pi / 2, Mr)
#
#         for j, p in reversed(list(enumerate(phi))):
#             for i, q in enumerate(self.theta):
#                 B = np.asarray([
#                     self.finger_props[4] * np.sin(p) * np.cos(q),
#                     self.finger_props[4] * np.sin(p) * np.sin(q),
#                     self.finger_props[3] + self.finger_props[4] * np.cos(p),
#                 ])
#
#                 B2 = np.asarray([
#                     self.finger_props[4] * G * np.sin(p) * np.cos(q),
#                     self.finger_props[4] * G * np.sin(p) * np.sin(q),
#                     self.finger_props[3] + self.finger_props[4] * G * np.cos(p),
#                 ])
#
#                 Ry = rotation_matrix(p, yaxis)
#                 Rz = rotation_matrix(q, zaxis)
#                 Rt = rotation_matrix(np.pi / 2, yaxis)
#                 Rx = rotation_matrix(np.pi, xaxis)
#
#                 rot = concatenate_matrices(Rz, Ry, Rt)[:3, :3]
#
#                 rot = R.from_matrix(rot).as_quat()
#
#                 self.push_points_end.append([[B[0], B[1], B[2]], rot.tolist()])
#                 self.push_points_start.append([[B2[0], B2[1], B2[2]], rot.tolist()])
#
#         # # display
#         if display:
#             from matplotlib import pyplot as plt
#             fig = plt.figure(figsize=(15, 15))
#             ax = fig.add_subplot(111, projection='3d')
#             for x in range(len(self.push_points_start)):
#                 ax.plot(*self.push_points_end[x][0], 'go')
#                 ax.plot(*self.push_points_start[x][0], 'ro')
#
#             plt.draw()
#             plt.show()
#
#     def get_push_point(self, q: float, h: float, i: int) -> Any:
#         '''Get start and end points for every press process
#
#         Parameters
#         ----------
#         q : float
#             angle id
#         h : float
#             hight id
#         i : int
#             _description_
#
#         Returns
#         -------
#         Any
#             push_point_start : list
#                 starting point for press process [xyz]
#             push_point_end : list
#                 end point for press process [xyz]
#         '''
#
#         G = 2
#         if i <= self.M // 2:
#             H = np.asarray([
#                 self.finger_props[0] + self.finger_props[4] * np.cos(q),
#                 self.finger_props[1] + self.finger_props[4] * np.sin(q),
#                 self.finger_props[2] + h,
#             ])
#
#             H2 = np.asarray([
#                 self.finger_props[0] + self.finger_props[4] * G * np.cos(q),
#                 self.finger_props[1] + self.finger_props[4] * G * np.sin(q),
#                 self.finger_props[2] + h,
#             ])
#
#             Rz = rotation_matrix(q, zaxis)
#             Rt = rotation_matrix(np.pi, zaxis)
#             Rx = rotation_matrix(np.pi, xaxis)
#
#             rot = concatenate_matrices(Rz, Rt, Rx)[:3, :3]
#
#             rot = R.from_matrix(rot[:3, :3]).as_quat()
#
#             push_point_end = [[H[0], H[1], H[2]], rot.tolist()]
#             push_point_start = [[H2[0], H2[1], H2[2]], rot.tolist()]
#
#         else:
#             phi = np.linspace(0, np.pi / 2, self.M)[::-1][i - 1]
#
#             B = np.asarray([
#                 self.finger_props[4] * np.sin(phi) * np.cos(q),
#                 self.finger_props[4] * np.sin(phi) * np.sin(q),
#                 self.finger_props[3] + self.finger_props[4] * np.cos(phi),
#             ])
#
#             B2 = np.asarray([
#                 self.finger_props[4] * G * np.sin(phi) * np.cos(q),
#                 self.finger_props[4] * G * np.sin(phi) * np.sin(q),
#                 self.finger_props[3] + self.finger_props[4] * G * np.cos(phi),
#             ])
#
#             Ry = rotation_matrix(phi, yaxis)
#             Rz = rotation_matrix(q, zaxis)
#             Rt = rotation_matrix(np.pi / 2, yaxis)
#             Rx = rotation_matrix(np.pi, xaxis)
#
#             rot = concatenate_matrices(Rz, Ry, Rt)[:3, :3]
#
#             rot = R.from_matrix(rot).as_quat()
#
#             push_point_end = [[B[0], B[1], B[2]], rot.tolist()]
#             push_point_start = [[B2[0], B2[1], B2[2]], rot.tolist()]
#
#         return push_point_start, push_point_end
#
#     def render_ball_at(self, angle: float, h: float, point: int) -> np.ndarray:
#         '''Takes the image from the sensor so that the sphere
#         is at the desired point depending on the given parameters
#
#         Parameters
#         ----------
#         angle : float
#             angle id (a.k.a 'q')
#         h : float
#             hight id
#         point : int
#             _description_ (a.k.a 'i')
#
#         Returns
#         -------
#         np.ndarray
#             color image
#         '''
#         # get the start and end points
#         push_point_start, push_point_end = self.get_push_point(angle, h + 0.016, point)
#
#         self._obj_x = push_point_start[0][0]
#         self._obj_y = push_point_start[0][1]
#         self._obj_z = push_point_start[0][2]
#
#         # reset the sphere pos to the start pos
#         pyb.resetBasePositionAndOrientation(self.obj.id, [self._obj_x, self._obj_y, self._obj_z], [0, 0, 0, 1])
#
#         cid = pyb.createConstraint(
#             self.obj.id,  # parent body unique id
#             -1,  # parent link index (or -1 for the base)
#             -1,  # child body unique id, or -1 for no body (specify anon-dynamic child frame in world coordinates)
#             -1,  # child link index, or -1 for the base
#             pyb.JOINT_FIXED,  # joint type: JOINT_PRISMATIC, JOINT_FIXED,JOINT_POINT2POINT, JOINT_GEAR
#             [0, 0, 0],  # joint axis, in child link frame
#             [0, 0, 0],  # position of the joint frame relative to parent center of mass frame.
#             [self._obj_x, self._obj_y, self._obj_z]
#             # position of the joint frame relative to a given child center of mass frame (or world origin if no child specified)
#         )
#
#         # render. need to check where to put this line
#         color, depth = self.allsight.render()  # depth = gel deformation [meters]
#
#         # if you want to see the sensor camera visualization - uncomment the row below
#         self.allsight.updateGUI(color, depth)
#
#         # if you want to debug the sphere pos - uncomment the rows below
#         # pose = p.getBasePositionAndOrientation(self.obj.id)[0][:3]
#         # print(pose)
#
#         self._obj_x = push_point_end[0][0]
#         self._obj_y = push_point_end[0][1]
#         self._obj_z = push_point_end[0][2]
#
#         pyb.changeConstraint(cid, [self._obj_x, self._obj_y, self._obj_z], maxForce=80)
#         # time.sleep(0.05)
#
#         color, depth = self.allsight.render()  # depth = gel deformation [meters]
#
#         self.allsight.updateGUI(color, depth)
#
#         return color[0]
