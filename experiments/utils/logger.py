import cv2
import os
from datetime import datetime
import json

import numpy as np
import pathlib


pc_name = os.getlogin()

class DataSimLogger():
    """
    Logs simulated data including images and metadata to disk.

    Attributes:
        data_dict (dict): Dictionary to store image metadata.
        img_press_dict (dict): Dictionary to store processed RGB images.
        depth_press_dict (dict): Dictionary to store processed depth images.
        save (bool): Flag indicating whether to save images.
        save_depth (bool): Flag indicating whether to save depth images.
        date (str): Current date and time formatted as "%Y_%m_%d-%I_%M_%S".
        dataset_path_images_rgb (str): Path to save RGB images.
        dataset_path_images_depth (str): Path to save depth images.
        dataset_path_data (str): Path to save metadata JSON files.

    Methods:
        __init__(self, prefix, leds, indenter, save=True, save_depth=False):
            Initializes the DataSimLogger instance.
        append(self, prefix, i, q, frame, depth, trans, rot, ft, contact_px, count):
            Appends data and images to the logger.
        save_batch_images(self):
            Saves all stored images to disk.
        save_data_dict(self):
            Saves the metadata dictionary to a JSON file.
    """

    def __init__(self,prefix, leds, indenter, save=True, save_depth=False):
        """
        Initializes the DataSimLogger instance.

        Args:
            prefix (str or None): Prefix for dataset folder names.
            leds (str): LED configuration identifier.
            indenter (str): Indenter identifier.
            save (bool, optional): Flag to save images. Defaults to True.
            save_depth (bool, optional): Flag to save depth images. Defaults to False.
        """

        self.data_dict = {}
        self.img_press_dict = {}
        self.depth_press_dict = {}
        self.save = save
        self.save_depth = save_depth
        # Init of the dataset dir paths with the current day and time
        self.date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
        if prefix is not None:
            self.dataset_path_images = "allsight_sim_dataset/clear/{}/images/{}/{}_img_{}".format(leds, indenter,prefix, self.date)
        else:
            self.dataset_path_images = "allsight_sim_dataset/clear/{}/images/{}/img_{}".format(leds, indenter, self.date)
        self.dataset_path_images_rgb = self.dataset_path_images
        self.dataset_path_images_depth = os.path.join(self.dataset_path_images,"depth")

        if prefix is not None:
            self.dataset_path_data = "allsight_sim_dataset/clear/{}/data/{}/{}_data_{}".format(leds, indenter,prefix, self.date)
        else:
            self.dataset_path_data = "allsight_sim_dataset/clear/{}/data/{}/data_{}".format(leds, indenter, self.date)
        

        if save:
            if not os.path.exists(self.dataset_path_images_rgb): pathlib.Path(self.dataset_path_images_rgb).mkdir(parents=True, exist_ok=True)
            
            if self.save_depth:
                if not os.path.exists(self.dataset_path_images_depth): pathlib.Path(self.dataset_path_images_depth).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(self.dataset_path_data): pathlib.Path(self.dataset_path_data).mkdir(parents=True, exist_ok=True)

    def append(self,prefix, i, q, frame, depth, trans, rot, ft,contact_px, count):
        """
        Appends data and images to the logger.

        Args:
            prefix (str or None): Prefix for image and depth IDs.
            i (int): Index of the image.
            q (float): Theta value.
            frame (np.ndarray): RGB image frame.
            depth (np.ndarray): Depth image frame.
            trans (tuple): Pose transformation (translation).
            rot (tuple): Pose transformation (rotation).
            ft (float): Frame timestamp.
            contact_px (int): Contact pixel value.
            count (int): Frame count.
        """

        if prefix is not None:
            img_id = '{}_image{}_{}_{:.2f}.jpg'.format(prefix,count, i, q)
        else:
            img_id = 'image{}_{}_{:.2f}.jpg'.format(count, i, q)
        img_path = os.path.join(self.dataset_path_images_rgb, img_id)

        if prefix is not None:
            depth_id = '{}_depth{}_{}_{:.2f}.jpg'.format(prefix,count , i, q)
        else:
            depth_id = 'depth{}_{}_{:.2f}.jpg'.format(count , i, q)
        depth_path = os.path.join(self.dataset_path_images_depth, depth_id)

        self.img_press_dict[img_path] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.depth_press_dict[depth_path] = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        contact_px = -1 if contact_px is None else contact_px

        self.data_dict[img_id] = {'frame': img_path,
                                  'depth': depth_path,
                                  'pose_transformed': (trans, rot),
                                  'ref_frame': os.path.join(self.dataset_path_images_rgb, 'ref_frame.jpg'),
                                  'theta': q,
                                  'ft': ft,
                                  'contact_px': (contact_px),
                                  'time': count,
                                  'num': i}

    def save_batch_images(self):
        """
        Saves all stored images to disk.
        """
        for key in self.img_press_dict.keys():

            if not cv2.imwrite(key, self.img_press_dict[key]):
                raise Exception("Could not write image")

        if self.save_depth:
            for key in self.depth_press_dict.keys():
                if not cv2.imwrite(key, self.depth_press_dict[key]):
                    raise Exception("Could not write image")
                
        # Clear the dict
        self.img_press_dict.clear()
        self.depth_press_dict.clear()

    def save_data_dict(self):
        """
        Saves the metadata dictionary to a JSON file.
        """

        path = os.path.join(self.dataset_path_data, 'data_{}.json'.format(self.date))
        if self.save:
            with open(path, 'w') as json_file:
                json.dump(self.data_dict, json_file, indent=3)