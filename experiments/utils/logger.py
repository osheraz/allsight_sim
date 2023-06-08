import cv2
import os
from datetime import datetime
import json

import numpy as np
import pathlib


pc_name = os.getlogin()

class DataSimLogger():

    def __init__(self, leds, indenter, save=True, save_depth=False):

        self.data_dict = {}
        self.img_press_dict = {}
        self.depth_press_dict = {}
        self.save = save
        self.save_depth = save_depth
        # Init of the dataset dir paths with the current day and time
        self.date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S")
        self.dataset_path_images = "dataset/{}/images/{}/img_{}".format(leds, indenter, self.date)
        self.dataset_path_images_rgb = self.dataset_path_images
        self.dataset_path_images_depth = os.path.join(self.dataset_path_images,"depth")

        self.dataset_path_data = "dataset/{}/data/{}/data_{}".format(leds, indenter, self.date)
        

        if save:
            if not os.path.exists(self.dataset_path_images_rgb): pathlib.Path(self.dataset_path_images_rgb).mkdir(parents=True, exist_ok=True)
            
            if self.save_depth:
                if not os.path.exists(self.dataset_path_images_depth): pathlib.Path(self.dataset_path_images_depth).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(self.dataset_path_data): pathlib.Path(self.dataset_path_data).mkdir(parents=True, exist_ok=True)

    def append(self, i, q, frame, depth, trans, rot, ft, count):

        img_id = 'image{}_{:.2f}_{:.2f}.jpg'.format(i, q, count)
        img_path = os.path.join(self.dataset_path_images_rgb, img_id)

        depth_id = 'depth{}_{:.2f}_{:.2f}.jpg'.format(i, q, count)
        depth_path = os.path.join(self.dataset_path_images_depth, depth_id)

        self.img_press_dict[img_path] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.depth_press_dict[depth_path] = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        self.data_dict[img_id] = {'frame': img_path,
                                  'depth': depth_path,
                                  'pose': (trans, rot),
                                  'theta': q,
                                  'ft': ft,
                                  'time': count,
                                  'num': i}

    def save_batch_images(self):
        # Save images
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

        path = os.path.join(self.dataset_path_data, 'data_{}.json'.format(self.date))
        if self.save:
            with open(path, 'w') as json_file:
                json.dump(self.data_dict, json_file, indent=3)