import cv2
import os
from datetime import datetime
import json

import numpy as np
import pathlib
from .util.util import tensor2im, foreground,inv_foreground, circle_mask

pc_name = os.getlogin()

class DataSimLogger():

    def __init__(self,prefix, leds, indenter, save=True, save_depth=False,ref_frame=None,  transform=None, model_G = None, device=None ):
        
        self.data_dict = {}
        self.img_press_dict = {}
        self.depth_press_dict = {}
        self.gan_press_dict = {}
        self.save = save
        self.save_depth = save_depth
        self.model_G = model_G
        self.transform = transform
        self.ref_frame = ref_frame
        self.device = device
        
        # Init of the dataset dir paths with the current day and time
        self.date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
        if prefix is not None:
            self.dataset_path_images = "allsight_sim_dataset/clear/{}/images/{}/{}_img_{}".format(leds, indenter,prefix, self.date)
        else:
            self.dataset_path_images = "allsight_sim_dataset/clear/{}/images/{}/img_{}".format(leds, indenter, self.date)
        self.dataset_path_images_rgb = self.dataset_path_images
        self.dataset_path_images_depth = os.path.join(self.dataset_path_images,"depth")
        self.dataset_path_images_gan = os.path.join(self.dataset_path_images,"gan")
        
        if prefix is not None:
            self.dataset_path_data = "allsight_sim_dataset/clear/{}/data/{}/{}_data_{}".format(leds, indenter,prefix, self.date)
        else:
            self.dataset_path_data = "allsight_sim_dataset/clear/{}/data/{}/data_{}".format(leds, indenter, self.date)


        if save:
            if not os.path.exists(self.dataset_path_images_rgb): pathlib.Path(self.dataset_path_images_rgb).mkdir(parents=True, exist_ok=True)
            
            
            if self.save_depth:
                if not os.path.exists(self.dataset_path_images_depth): pathlib.Path(self.dataset_path_images_depth).mkdir(parents=True, exist_ok=True)

            if self.model_G is not None:
                if not os.path.exists(self.dataset_path_images_gan): pathlib.Path(self.dataset_path_images_gan).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(self.dataset_path_data): pathlib.Path(self.dataset_path_data).mkdir(parents=True, exist_ok=True)

        
        
    def append(self,prefix, i, q, frame, depth, trans, rot, ft,contact_px, count):

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

        if self.model_G is not None:
            if prefix is not None:
                gan_id = '{}_gan{}_{}_{:.2f}.jpg'.format(prefix,count , i, q)
            else:
                gan_id = 'gan{}_{}_{:.2f}.jpg'.format(count , i, q)
            gan_path = os.path.join(self.dataset_path_images_gan, gan_id)
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img_press_dict[img_path] = rgb_image
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
        
        if self.model_G is not None:
            color_tensor = self.transform(foreground(rgb_image,self.ref_frame)).unsqueeze(0).to(self.device)
            gan_image = inv_foreground(self.ref_frame,tensor2im(self.model_G(color_tensor)))
            self.gan_press_dict[gan_path] = gan_image
            self.data_dict[img_id].update({'gan':gan_path})


    def save_batch_images(self):
        # Save images
        print("Saving batch of images...")
        for key in self.img_press_dict.keys():
            if not cv2.imwrite(key, self.img_press_dict[key]):
                raise Exception("Could not write image")

        if self.save_depth:
            for key in self.depth_press_dict.keys():
                if not cv2.imwrite(key, self.depth_press_dict[key]):
                    raise Exception("Could not write image")
        
        
        if self.model_G is not None:
            for key in self.gan_press_dict.keys():
                if not cv2.imwrite(key, self.gan_press_dict[key]):
                    raise Exception("Could not write image")

        # Clear the dict
        self.img_press_dict.clear()
        self.depth_press_dict.clear()
        self.gan_press_dict.clear()

    def save_data_dict(self):

        path = os.path.join(self.dataset_path_data, 'data_{}.json'.format(self.date))
        if self.save:
            with open(path, 'w') as json_file:
                json.dump(self.data_dict, json_file, indent=3)