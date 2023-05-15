'''
TODO:
- check render_ball_at function
- check speed of algo
-
'''

import logging
import sys
from datetime import datetime
import os
import hydra
import re
import hyperopt
from hyperopt import fmin, tpe

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import time

# import allsight wrapper
PATH = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, PATH)
from tacto_allsight_wrapper.allsight_simulator import Simulator
from tacto_allsight_wrapper.allsight_wrapper import circle_mask, Sensor
from calibration.search_space import get_search_space

log = logging.getLogger(__name__)
origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)  # delete?


class Calibration():
    '''Calibration class for handling with the sim-real conf calibration process
    '''

    def __init__(self, simulator: Simulator, calib_folder: str = 'calibration/imgs0') -> None:
        '''Initialize calibration object

        Parameters
        ----------
        simulator : Simulator
            allsight simulator object
        calib_folder : str, optional
            real images folder for calibration, by default 'calibration/imgs0'
        '''
        self.simulator = simulator

        # get search space
        self.search_space = get_search_space()
        self.batch_size = 5

        # relative path
        self.calib_folder = os.path.join(PATH, calib_folder)

        # takes the path of the background image and a sorted list of the rest paths
        bg_path, self.real_image_paths = self._init_real_image_paths()
        bg = cv2.imread(bg_path)

        # set background image for the sensor readings
        self.simulator.allsight.renderer.set_background(bg)

    def _init_real_image_paths(self):
        '''takes the paths of the images from calib folder

        Returns
        -------
        bg_path: path of the background image
        real_image_paths: sorted list of iamges paths
        '''

        real_image_paths = sorted(os.listdir(self.calib_folder))
        bg_path = os.path.join(self.calib_folder, real_image_paths[0])
        real_image_paths = real_image_paths[1:]
        return bg_path, real_image_paths

    def objective_batch(self, params: dict) -> dict:
        '''objective function to minimize using hyperopt (batch mode)

        Parameters
        ----------
        params : dict
            hyperparameters

        Returns
        -------
        dict
            loss
            status
            concat_images - image of real vs sim 
        '''

        # takes the number of batch depend on the num of images
        num_real_images = len(self.real_image_paths)
        num_batches = int(np.ceil(num_real_images / self.batch_size))

        # store real images in batches and apply masking
        real_image_batches = []
        real_image_batches_paths = []
        for i in range(num_batches):
            batch_paths = self.real_image_paths[i * self.batch_size:(i + 1) * self.batch_size]
            real_image_batches_paths.append(batch_paths)

            batch_images = []
            for path in batch_paths:
                img = cv2.imread(os.path.join(self.calib_folder, path))
                img[circle_mask() == 0] = 0
                batch_images.append(np.array(img))
            real_image_batches.append(np.stack(batch_images))

        # Generate simulated images batches and calculate loss
        total_loss = 0
        for i, real_batch in enumerate(real_image_batches):
            simulated_batch = self.generate_simulated_images(params, real_image_batches_paths[i])
            batch_loss = self.calculate_batch_loss(simulated_batch, real_batch)

            total_loss += batch_loss

            real_image = real_batch[0]
            simulate_image = simulated_batch[0]
            concatenate_images = np.concatenate((real_image, simulate_image))
            cv2.imshow('concat', concatenate_images)
            cv2.waitKey(1)

        # take the avg batches loss
        avg_loss = total_loss / num_batches

        # for debug uncomment the rows below
        # print(f"params: {params}")
        # print(f"loss: {avg_loss}")

        return {'loss': avg_loss, 'status': 'ok',
                # -- store other results like this
                'concat_imgs': concatenate_images}

    def generate_simulated_images(self, params: dict, real_batch_paths: list) -> np.ndarray:
        '''create simualted images batches depend on the params and real iamges properties

        Parameters
        ----------
        params : dict
            conf hyperparameters
        real_batch_paths : list
            real images paths list

        Returns
        -------
        np.ndarray
            simulated iamges batch
        '''

        ## here you should init all the hyperparameters into the simulation object        
        self.simulator.allsight.renderer.conf.sensor.camera[0]['position'][0] = float(params['camera']['position'])
        self.simulator.allsight.renderer.conf.sensor.camera[0]['yfov'] = float(params['camera']['yfov'])
        self.simulator.allsight.renderer._init_camera()

        # create simulated batch
        simulated_batch = []
        for img_path in real_batch_paths:
            # decode the image properties from the path format
            push_pose_params = self.path_decode(img_path)
            q_normalized = push_pose_params['q']
            q = np.interp(q_normalized, [0, 1], [0, 2 * np.pi])
            i = push_pose_params['i']
            h = self.simulator.hight_presses[i]

            # get the the specific image needed
            simulate_image = self.simulator.render_ball_at(q, h, i)
            simulate_image = cv2.cvtColor(simulate_image, cv2.COLOR_BGR2RGB)
            simulated_batch.append(simulate_image)

        return np.stack(simulated_batch)

    @staticmethod
    def path_decode(path: str) -> dict:
        '''decode path str to the relevant params

        Parameters
        ----------
        path : str
            iamge path

        Returns
        -------
        dict
            iamge press point params
        '''
        numbers_st = re.findall(r'[+-]?\d+(?:\.\d+)?', path)
        numbers_fl = [float(num) for num in numbers_st]
        decoded_dic = {'i': int(numbers_fl[0]), 'q': numbers_fl[1], 'time': numbers_fl[2]}

        return decoded_dic

    def objective(self, params: dict) -> dict:
        '''objective function to minimize using hyperopt (single mode)

        Parameters
        ----------
        params : dict
            hyperparameters

        Returns
        -------
        dict
            loss
            status
            concat_images - image of real vs sim 
        '''

        self.simulator.allsight.renderer.conf.sensor.camera[0]['position'][0] = float(params['camera']['position'])
        self.simulator.allsight.renderer.conf.sensor.camera[0]['yfov'] = float(params['camera']['yfov'])
        self.simulator.allsight.renderer._init_camera()

        print(params)

        # decode the image properties from the path format 
        push_pose_params = self.path_decode('image3_0.43_5.12.jpeg')
        q_normalized = push_pose_params['q']
        q = np.interp(q_normalized, [0, 1], [0, 2 * np.pi])
        i = push_pose_params['i']
        h = self.simulator.hight_presses[i]

        simulate_image = self.simulator.render_ball_at(q, h, i)
        simulate_image = cv2.cvtColor(simulate_image, cv2.COLOR_BGR2RGB)

        real_image = cv2.imread(PATH + '/calibration/imgs/image3_0.43_5.12.jpeg')
        real_image[circle_mask() == 0] = 0

        ssim_score = ssim(real_image, simulate_image, multichannel=True)
        loss = 1 - ssim_score
        print(f"loss: {loss}")

        concatenate_images = np.concatenate((real_image, simulate_image))
        cv2.imshow('concat', concatenate_images)
        cv2.waitKey(1)

        return {'loss': loss, 'status': 'ok', 'concat_imgs': concatenate_images}

    def calculate_batch_loss(self, real_batch: np.ndarray, sim_batch: np.ndarray) -> float:
        '''calculate ssim loss between real and sim images batches

        Parameters
        ----------
        real_batch : np.ndarray
            batch of real images
        sim_batch : np.ndarray
            batch of sim images

        Returns
        -------
        float
            ssim loss
        '''
        batch_ssim = []

        for real_img, sim_img in zip(real_batch, sim_batch):
            real_img = real_img.astype('float32')
            sim_img = sim_img.astype('float32')

            # take the 1-ssim for minimizing 
            loss = 1 - ssim(real_img, sim_img, multichannel=True)

            batch_ssim.append(loss)

        return np.mean(batch_ssim)


# Load the config YAML file from experiments/conf/allsight.yaml
@hydra.main(config_path="conf", config_name="allsight")
def calibration(cfg):
    # start script from the path of the paraent dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # create simulator object
    sim = Simulator(cfg,
                    with_bg=True,
                    attributes={'M': 10, 'N': 29})

    # create env
    sim.create_env(cfg, obj_id='20')

    # start sim thread
    sim.start()

    # create calibration object
    calib = Calibration(sim)

    # apply hyperopt optimization
    # NOTE: you can use objective or objective_batch  
    best_params = fmin(
        # calib.objective_batch,
        calib.objective_batch,
        space=calib.search_space,
        algo=tpe.suggest,
        max_evals=20,
    )

    # apply the objective on the best params and show concat images
    # res = calib.objective_batch({'camera':best_params})
    res = calib.objective_batch({'camera': best_params})
    concatenate_images = res['concat_imgs']
    print(best_params)
    cv2.imshow('result', concatenate_images)
    cv2.waitKey(0)


def main():
    calibration()


if __name__ == '__main__':
    main()
