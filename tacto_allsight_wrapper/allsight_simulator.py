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
from experiments.utils.geometry import rotation_matrix, concatenate_matrices, convert_quat_xyzw_to_wxyz, convert_quat_wxyz_to_xyzw
from scipy.spatial.transform import Rotation as R
from transformations import translation_matrix, translation_from_matrix, quaternion_matrix, quaternion_from_matrix

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
        bg = cv2.imread(os.path.join(PATH, f"experiments/conf/ref/ref_frame_{leds}.jpg"))
        conf_path = os.path.join(PATH, f"experiments/conf/sensor/config_allsight_{leds}.yml")

        # initialize allsight
        self.allsight = allsight_wrapper.Sensor(
            **cfg.tacto, **{"config_path": conf_path},
            background=bg if with_bg else None
        )

        self.base_h = cfg.sensor_dims.base_h
        self.cyl_h = cfg.sensor_dims.cyl_h
        self.cyl_r = cfg.sensor_dims.cyl_r

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

        # pyb.connect(pyb.DIRECT)
        # pyb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth

        self.body = px.Body(**cfg.allsight)

        # object body 
        # take the urdf path with the relevant id given
        if obj_id in ['sphere3', 'sphere4', 'sphere5']:
            obj_urdf_path = f"../assets/objects/{obj_id}.urdf"
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

        self.cyl_split = conf['cyl_split']
        self.top_split = conf['top_split']
        self.angle_split = conf['angle_split']

        # Create constraints
        '''
        due to limitation of pybullet orientation handling,
        instead of changing the constraint, we recreate it every timestamp
        '''
        # (self._obj_x, self._obj_y, self._obj_z), self._obj_or = self.obj.get_base_pose()
        # init_xyz = [self._obj_x, self._obj_y, self._obj_z]

        # self.cid = pyb.createConstraint(
        #     self.obj.id,  # parent body unique id
        #     -1,  # parent link index (or -1 for the base)
        #     -1,  # child body unique id, or -1 for no body (specify anon-dynamic child frame in world coordinates)
        #     -1,  # child link index, or -1 for the base
        #     pyb.JOINT_FIXED,  # joint type: JOINT_PRISMATIC, JOINT_FIXED,JOINT_POINT2POINT, JOINT_GEAR
        #     [0, 0, 0],  # joint axis, in child link frame
        #     [0, 0, 0],  # position of the joint frame relative to parent center of mass frame.
        #     init_xyz
        #     # position of the joint frame relative to a given child center of mass frame (or world origin if no child specified)
        # )

        # create data logger object
        self.logger = DataSimLogger(conf["save_prefix"],
                                    conf['leds'],
                                    conf['indenter'],
                                    save=conf['save'],
                                    save_depth=False)

        # take ref frame
        ref_frame, _ = self.allsight.render()

        ref_img_color_path = os.path.join(self.logger.dataset_path_images, 'ref_frame.jpg')

        if conf['save']:
            ref_frame[0] = cv2.cvtColor(ref_frame[0], cv2.COLOR_BGR2RGB)
            if not cv2.imwrite(ref_img_color_path, ref_frame[0]):
                raise Exception("Could not write image")

        frame_count = 0

        Q = np.linspace(0, 2 * np.pi, conf['angle_split'])
        current_pos, current_quat = pyb.getBasePositionAndOrientation(self.body.id)
        current_euler = pyb.getEulerFromQuaternion(current_quat)

        for q in Q:

            for i in range(conf['start_from'], self.cyl_split + self.top_split, 1):

                removed = False
                if i == self.cyl_split: continue

                # Calculate the new orientation by adding the desired rotation
                new_euler = [current_euler[0], current_euler[1], current_euler[2] + q]
                new_quat = pyb.getQuaternionFromEuler(new_euler)

                self.body.set_base_pose(position=current_pos, orientation=new_quat)
                pyb.stepSimulation()

                push_point_start, push_point_end = self.get_push_point_by_index(0, i)

                # pyb.changeConstraint(self.cid, jointChildPivot=push_point_start[0],
                #                      jointChildFrameOrientation=push_point_start[1],
                #                      maxForce=f_push)

                self.cid = pyb.createConstraint(self.obj.id, -1, -1, -1, pyb.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                childFramePosition=push_point_start[0],
                                                childFrameOrientation=push_point_start[1])

                # self.obj.set_base_pose(position=push_point_start[0], orientation=push_point_start[1])

                color, depth = self.allsight.render()
                self.allsight.updateGUI(color, depth)
                time.sleep(0.1)

                pyb.removeConstraint(self.cid)

                # pyb.changeConstraint(self.cid, jointChildPivot=push_point_end[0],
                #                      jointChildFrameOrientation=push_point_end[1],
                #                      maxForce=f_push)

                self.cid = pyb.createConstraint(self.obj.id, -1, -1, -1, pyb.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                childFramePosition=push_point_end[0],
                                                childFrameOrientation=push_point_end[1])

                # self.obj.set_base_pose(position=push_point_end[0], orientation=push_point_end[1])

                for _ in range(5):

                    color, depth = self.allsight.render()
                    time.sleep(0.05)

                    if np.sum(depth):
                        self.allsight.updateGUI(color, depth)

                        # time.sleep(0.05)
                        pose = list(pyb.getBasePositionAndOrientation(self.obj.id)[0][:3])

                        pose[0] -= np.sign(pose[0]) * 0.002  # radi
                        pose[1] -= np.sign(pose[1]) * 0.002  # radi
                        pose[2] -= self.base_h + 0.006       # base_h

                        rot = pyb.getBasePositionAndOrientation(self.obj.id)[1][:4]

                        trans_mat, rot_mat = translation_matrix(pose), quaternion_matrix(convert_quat_xyzw_to_wxyz(rot))
                        T_finger_origin_press = np.dot(trans_mat, rot_mat)
                        T_finger_origin_press_rotate_q = np.matmul(rotation_matrix(q, zaxis), T_finger_origin_press)
                        pose = translation_from_matrix(T_finger_origin_press_rotate_q).tolist()
                        rot = convert_quat_wxyz_to_xyzw(quaternion_from_matrix(T_finger_origin_press_rotate_q).tolist())

                        force = self.allsight.get_force('cam0')['2_-1']

                        color_img = color[0]
                        depth_img = np.concatenate(list(map(self.allsight._depth_to_color, depth)), axis=1)

                        self.logger.append(conf["save_prefix"], i, np.interp(q, [0, 2 * np.pi], [0, 1]),
                                           color_img, depth_img, pose, rot, force, frame_count)

                        frame_count += 1
                        pyb.removeConstraint(self.cid)
                        removed = True
                        break

                if not removed:
                    pyb.removeConstraint(self.cid)
                    removed = True

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

        H_cyl = np.linspace(0, self.cyl_h, self.cyl_split)

        x0, y0 = 0, 0
        if i < len(H_cyl):

            H = np.asarray([
                x0 + self.cyl_r * np.cos(q),
                y0 + self.cyl_r * np.sin(q),
                self.base_h + H_cyl[i],
            ])

            H2 = np.asarray([
                x0 + self.cyl_r * G * np.cos(q),
                y0 + self.cyl_r * G * np.sin(q),
                self.base_h + H_cyl[i],
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

            B = np.asarray([
                self.cyl_r * np.sin(phi) * np.cos(q),
                self.cyl_r * np.sin(phi) * np.sin(q),
                self.base_h + self.cyl_h + self.cyl_r * np.cos(phi),
            ])

            B2 = np.asarray([
                self.cyl_r * G * np.sin(phi) * np.cos(q),
                self.cyl_r * G * np.sin(phi) * np.sin(q),
                self.base_h + self.cyl_h + self.cyl_r * G * np.cos(phi),
            ])

            Ry = rotation_matrix(phi, yaxis)
            Rz = rotation_matrix(q, zaxis)
            Rt = rotation_matrix(np.pi / 2, yaxis)

            rot = concatenate_matrices(Rz, Ry, Rt)[:3, :3]

            rot = R.from_matrix(rot).as_quat()

            push_point_end = [[B[0], B[1], B[2]], rot.tolist()]
            push_point_start = [[B2[0], B2[1], B2[2]], rot.tolist()]

        return push_point_start, push_point_end

    def update_pose_given_point(self, point, press_depth, shear_mag, delta=0):
        """
        Convert meter to pixels
        """
        sensor_vertices = self.allsight.renderer.sensor_vertices
        sensor_normals = self.allsight.renderer.sensor_normals

        dist = np.linalg.norm(point - sensor_vertices, axis=1)
        idx = np.argmin(dist)

        # idx: the idx vertice, get a new pose
        new_position = sensor_vertices[idx].copy()
        new_orientation = sensor_normals[idx].copy()

        delta = np.random.uniform(low=0.0, high=2 * np.pi, size=(1,))[0]

        from contrib.pose import pose_from_vertex_normal

        new_pose = pose_from_vertex_normal(
            new_position, new_orientation, shear_mag, delta
        ).squeeze()
        self.update_pose_given_pose(press_depth, new_pose)

    def update_pose_given_pose(self, press_depth, obj_pos):
        """
        Given tf gel_pose and press_depth, update tacto camera
        """
        self.press_depth = press_depth
        press_pos = self.add_press(obj_pos)

        def matrix2quaternion(matrix):
            r = R.from_matrix(matrix[:3, :3])
            quaternion = r.as_quat()
            translation = matrix[:3, 3]
            return quaternion, translation

        ori, pos = matrix2quaternion(press_pos)
        self.obj.set_base_pose(position=pos, orientation=ori)

        # self.allsight.renderer.update_object_pose_from_matrix('2_-1', press_pos)

    def add_press(self, pose):
        """
        Add sensor penetration
        """
        pen_mat = np.eye(4)
        pen_mat[2, 3] = -self.press_depth
        return np.matmul(pose, pen_mat)