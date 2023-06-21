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
        self.allsight = allsight_wrapper.Sensor(
            **cfg.tacto, **{"config_path": conf_path},
            background=bg if with_bg else None
        )

        self.cyl_split = summary['cyl_split']
        self.top_split = summary['top_split']
        self.angle_split = summary['angle_split']

        # TODO should be constant
        self.start_h = 0.012
        self.finger_props = [0, 0, self.start_h, 0.016, 0.0128]

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
        if obj_id in ['20', '30', '25']:
            obj_urdf_path = f"../assets/objects/sphere_{obj_id}.urdf"
        elif obj_id in ['cube', 'rect', 'ellipse']:
            obj_urdf_path = f"../assets/objects/{obj_id}_small.urdf"

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

        Q = np.linspace(0, 2 * np.pi, conf['angle_split'])

        for q in Q:

            for i in range(conf['start_from'], self.cyl_split + self.top_split, 1):

                if i == self.cyl_split: continue

                # f_push = 80 if i < self.cyl_split else 70

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

                pose[0] -= np.sign(pose[0]) * 0.002  # radi
                pose[1] -= np.sign(pose[1]) * 0.002  # radi
                pose[2] -= self.start_h + 0.006

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

        G = 2
        H_cyl = np.linspace(0, self.finger_props[3], self.cyl_split)

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

            phi = np.linspace(0, np.pi / 2, self.top_split)[::-1][i - len(H_cyl)]

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

