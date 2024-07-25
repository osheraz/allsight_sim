import logging
import os
import cv2
import numpy as np
import pybullet as p
import pyrender
from omegaconf import OmegaConf, DictConfig
from scipy.spatial.transform import Rotation as R

from tacto import Renderer as tRenderer, Sensor as tSensor


# helper functions
def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=False):
    """
    Convert Euler angles to a transformation matrix.

    Args:
        angles (list, optional): Euler angles [theta_x, theta_y, theta_z]. Defaults to [0, 0, 0].
        translation (list, optional): Translation vector [tx, ty, tz]. Defaults to [0, 0, 0].
        xyz (str, optional): Order of Euler angles. Defaults to "xyz".
        degrees (bool, optional): Whether angles are in degrees. Defaults to False.

    Returns:
        np.ndarray: Transformation matrix.
    """
    r = R.from_euler(xyz, angles, degrees=degrees)
    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = r.as_matrix()
    return pose


def circle_mask(size=(480, 480), border=0):
    """
    Create a circular mask for filtering images.

    Args:
        size (tuple, optional): Size of the mask (width, height). Defaults to (480, 480).
        border (int, optional): Border thickness around the circle. Defaults to 0.

    Returns:
        np.ndarray: Circular mask as an image.
    """
    m = np.zeros((size[1], size[0]))
    m_center = (size[0] // 2, size[1] // 2)
    m_radius = min(size[0], size[1]) // 2 - border
    m = cv2.circle(m, m_center, m_radius, 255, -1)

    m /= 255
    m = m.astype(np.float32)
    mask = np.stack([m, m, m], axis=2)
    return mask


class Renderer(tRenderer):
    """
    allsight custom renderer

    Parameters:
        width : int
            Width of the renderer.
        height : int
            Height of the renderer.
        background : Any
            Background configuration.
        config_path : str
            Path to configuration file.
    """

    def __init__(self, width, height, background, config_path):
        """
        Initializes the Renderer.

        Args:
            width (int): Width of the renderer.
            height (int): Height of the renderer.
            background (Any): Background configuration.
            config_path (str): Path to configuration file.
        """
        super().__init__(width, height, background, config_path, headless=True)

    def _post_process(self, color, depth, camera_index, noise=True, calibration=True):
        """
        Post-processes rendered images.

        Args:
            color (np.ndarray): Color image.
            depth (np.ndarray): Depth image.
            camera_index (int): Index of the camera.
            noise (bool, optional): Whether to add noise. Defaults to True.
            calibration (bool, optional): Whether to calibrate. Defaults to True.

        Returns:
            tuple: Processed color and depth images.
        """
        if calibration:
            # use custom _calibration method for background image
            color = self._calibrate(color, camera_index, self.conf.sensor.bg_calibration)
        if noise:
            color = self._add_noise(color)

        return color, depth

    def _calibrate(self, color: np.ndarray, camera_index: int, cfg: DictConfig) -> np.ndarray:
        """
        Custom calibration method for background image.

        Args:
            color (np.ndarray): RGB background image.
            camera_index (int): Camera ID.
            cfg (DictConfig): Configuration for background calibration.

        Returns:
            np.ndarray: Calibrated background color image.
        """

        # init conf
        if cfg.enable:
            s_factor = cfg.scale_factor
            k_size = cfg.blur.k_size
            sig = cfg.blur.sigma
            clip = cfg.clip

            if self._background_real is not None:
                # Simulated difference image, with scaling factor 0.5
                diff = (color.astype(np.float) - self._background_sim[camera_index]) * s_factor

                # Add low-pass filter to match real readings
                diff = cv2.GaussianBlur(diff, (k_size, k_size), sig)

                # Combine the simulated difference image with real background image
                color = np.clip((diff[:, :, :3] + self._background_real), clip[0], clip[1]).astype(
                    np.uint8
                )

            return color

        # if not enable use tacto _calibrate method
        else:
            return super()._calibrate(color, camera_index)

    def _init_light(self, light=None):
        """
        Custom initialization method for light parameters.

        Args:
            light: Optional. Light parameters.

        Notes:
            Overrides parameters such as spotlight inner/outer cones from the config.
        """

        # Load light from config file
        if light is None:
            light = self.conf.sensor.lights

        origin = np.array(light.origin)

        xyz = []
        if light.polar:
            # Apply polar coordinates
            thetas = light.xrtheta.thetas
            rs = light.xrtheta.rs
            xs = light.xrtheta.xs
            for i in range(len(thetas)):
                theta = np.pi / 180 * thetas[i]
                xyz.append([xs[i], rs[i] * np.cos(theta), rs[i] * np.sin(theta)])
        else:
            # Apply cartesian coordinates
            xyz = np.array(light.xyz.coords)

        colors = np.array(light.colors)
        intensities = light.intensities

        # Save light nodes
        self.light_nodes = []
        self.light_poses0 = []

        for i in range(len(colors)):

            if not self.spot_light_enabled:
                # create pyrender.PointLight
                color = colors[i]
                light_pose_0 = euler2matrix(
                    angles=[0, 0, 0], translation=xyz[i] + origin
                )

                light = pyrender.PointLight(color=color, intensity=intensities[i])

            elif self.spot_light_enabled:

                # load spotlight angles from conf
                inner_angle = self.conf.sensor.lights.spot_angles.inner
                outer_angle = self.conf.sensor.lights.spot_angles.outer

                # create pyrender.SpotLight
                color = colors[i]

                theta = np.pi / 180 * (thetas[i] - 90)
                tuning_angle = -np.pi / 16
                light_pose_0 = euler2matrix(
                    xyz="yzx",
                    angles=[tuning_angle, 0, theta],
                    translation=xyz[i] + origin,
                )

                light = pyrender.SpotLight(
                    color=color,
                    intensity=intensities[i],
                    innerConeAngle=np.pi * inner_angle,
                    outerConeAngle=np.pi * outer_angle,
                )

            light_node = pyrender.Node(light=light, matrix=light_pose_0)

            self.scene.add_node(light_node)
            self.light_nodes.append(light_node)
            self.light_poses0.append(light_pose_0)
            self.current_light_nodes.append(light_node)

            # Add extra light node into scene_depth
            light_node_depth = pyrender.Node(light=light, matrix=light_pose_0)
            self.scene_depth.add_node(light_node_depth)


C_PATH = os.path.join(os.path.dirname(__file__)) + "../experiments/conf/sensor"


def _get_default_config(filename):
    """
    Get the default configuration file path.

    Args:
        filename (str): Configuration file name.

    Returns:
        str: Full path to the configuration file.
    """
    return os.path.join(C_PATH, filename)


def get_allsight_config_path():
    """
    Get the path to the AllSight configuration file.

    Returns:
        str: Path to the configuration file.
    """
    return _get_default_config("config_allsight.yml")


class Sensor(tSensor):
    """
    Sensor class for handling sensor functionalities.

    Parameters:
        width : int, optional
            Width of the sensor. Defaults to 120.
        height : int, optional
            Height of the sensor. Defaults to 160.
        background : Any, optional
            Background image. Defaults to None.
        config_path : str, optional
            Path to configuration file. Defaults to get_allsight_config_path().
        visualize_gui : bool, optional
            Whether to visualize GUI. Defaults to True.
        show_depth : bool, optional
            Whether to show depth. Defaults to True.
        zrange : float, optional
            Range of z-axis. Defaults to 0.002.
        cid : int, optional
            Camera ID. Defaults to 0.
    """
    def __init__(
            self,
            width=120,
            height=160,
            background=None,
            config_path=get_allsight_config_path(),
            visualize_gui=True,
            show_depth=True,
            zrange=0.002,
            cid=0,
    ):
        """
        Initializes the Sensor object.

        Args:
            width (int, optional): Width of the sensor. Defaults to 120.
            height (int, optional): Height of the sensor. Defaults to 160.
            background (Any, optional): Background image. Defaults to None.
            config_path (str, optional): Path to configuration file. Defaults to get_allsight_config_path().
            visualize_gui (bool, optional): Whether to visualize GUI. Defaults to True.
            show_depth (bool, optional): Whether to show depth. Defaults to True.
            zrange (float, optional): Range of z-axis. Defaults to 0.002.
            cid (int, optional): Camera ID. Defaults to 0.
        """
        self.cid = cid
        self.renderer = Renderer(width, height, background, config_path)

        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.zrange = zrange

        self.cameras = {}
        self.nb_cam = 0
        self.objects = {}
        self.object_poses = {}
        self.normal_forces = {}
        self._static = None
        self.blur_enabled = self.renderer.conf.sensor.blur_contact.enable

    def render(self):
        """
        Render tacto images from each camera's view.

        Returns:
            tuple: Color and depth images.
        """

        self._update_object_poses()

        colors = []
        depths = []

        for i in range(self.nb_cam):
            cam_name = "cam" + str(i)

            # get the contact normal forces
            normal_forces = self.get_force(cam_name)

            if normal_forces:
                position, orientation = self.cameras[cam_name].get_pose()
                self.renderer.update_camera_pose(position, orientation)
                color, depth = self.renderer.render(self.object_poses, normal_forces)

                # Remove the depth from curved gel
                for j in range(len(depth)):
                    depth[j] = self.renderer.depth0[j] - depth[j]
            else:
                color, depth = self._render_static()

            if self.blur_enabled:
                depth_map = np.concatenate(list(map(self._depth_to_color, depth)), axis=1)
                color = self._blur_contact(color, depth_map)

            mask = circle_mask(size=(480, 480))
            # mask = circle_mask()

            color[0][mask == 0] = 0

            colors += color
            depths += depth
            
        return colors, depths

    def _blur_contact(self, color: np.ndarray, depth_map: np.ndarray) -> list:
        """
        Blur contact area depending on the configuration.

        Args:
            color (np.ndarray): Color image.
            depth_map (np.ndarray): Depth map.

        Returns:
            list: List of blurred images.
        """
        # init blur conf from config_allsight.yaml
        blur = self.renderer.conf.sensor.blur_contact

        depth_map_g = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        mask = depth_map_g.copy()
        img = np.array(color[0])
        # split the image into channels
        b, g, r = cv2.split(img)

        # apply the mask to each channel
        b_masked = cv2.bitwise_and(b, b, mask=mask)
        g_masked = cv2.bitwise_and(g, g, mask=mask)
        r_masked = cv2.bitwise_and(r, r, mask=mask)

        # combine the masked channels back into an image
        masked_img = cv2.merge((b_masked, g_masked, r_masked))

        # invert the mask
        inv_mask = cv2.bitwise_not(mask)

        # apply a Gaussian blur to the inverted mask
        blurred_inv_mask = cv2.GaussianBlur(inv_mask, (blur.inv_mask_img.k_size, blur.inv_mask_img.k_size),
                                            blur.inv_mask_img.sigma)

        # create a mask for the blurred shape
        blurred_mask = cv2.bitwise_not(blurred_inv_mask)
        blurred_mask = cv2.cvtColor(blurred_mask, cv2.COLOR_GRAY2BGR)

        # split the blurred image into channels
        blurred_b, blurred_g, blurred_r = cv2.split(masked_img)

        # blur the masked channels using the blurred mask
        blurred_b = cv2.GaussianBlur(b_masked, (blur.mask_img.k_size, blur.mask_img.k_size), blur.mask_img.sigma)
        blurred_g = cv2.GaussianBlur(g_masked, (blur.mask_img.k_size, blur.mask_img.k_size), blur.mask_img.sigma)
        blurred_r = cv2.GaussianBlur(r_masked, (blur.mask_img.k_size, blur.mask_img.k_size), blur.mask_img.sigma)

        # combine the blurred channels back into an image
        blurred_img = cv2.merge((blurred_b, blurred_g, blurred_r))

        # combine the original image and the blurred image using the mask
        result = cv2.addWeighted(img, blur.add_weighted.w_real, blurred_img, blur.add_weighted.w_sim,
                                 blur.add_weighted.bias, dtype=cv2.CV_8U)

        return [result]

    def _subtract_bg(self, img1, img2, offset=0.5):
        """
        Subtract background image from the input image.

        Args:
            img1 (np.ndarray): Input image to be processed.
            img2 (np.ndarray): Background image.
            offset (float, optional): Offset value. Defaults to 0.5.

        Returns:
            np.ndarray: Resultant image after background subtraction.
        """

        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff

    def updateGUI(self, colors, depths, colors_gan=[], contact_px=None, bg=None):
        """
        Update images for visualization.

        Args:
            colors (list): List of color images.
            depths (list): List of depth images.
            colors_gan (list, optional): List of GAN-generated color images. Defaults to [].
            contact_px (list, optional): Contact pixel information. Defaults to None.
            bg (Any, optional): Background image. Defaults to None.
        """
        if not self.visualize_gui:
            return

        # concatenate colors horizontally (axis=1)
        color = np.concatenate(colors, axis=1)

        if len(colors_gan)!=0:
            color_gan = np.concatenate(colors_gan, axis=1)

        if bg is not None:
            mask = circle_mask()
            color = self._subtract_bg(color, bg[0]) * mask
            if len(colors_gan) != 0:
                color_gan = self._subtract_bg(color_gan, bg[0]) * mask

        if contact_px is not None:
            [x,y,r] = contact_px
            # Draw the circle on the original image
            cv2.circle(color, (x, y), int(r*2.5), (0, 255, 0), 4)
            # Draw a small circle at the center of the detected circle
            cv2.circle(color, (x, y), 2, (0, 0, 255), 3)

        if self.show_depth:
            # concatenate depths horizontally (axis=1)
            depth = np.concatenate(list(map(self._depth_to_color, depths)), axis=1)
            if bg is not None:
                color = cv2.cvtColor((color * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # concatenate the resulting two images vertically (axis=0)
            if len(colors_gan) == 0:
                color_n_depth = np.concatenate([color, depth], axis=0)
            else:
                color_n_depth = np.concatenate([color,color_gan, depth], axis=0)
            cv2.imshow(
                "color and depth", color_n_depth
            )
        else:
            cv2.imshow("color", color)

        cv2.waitKey(1)

    def detect_contact(self,depths)->list:
        """
        Detect contact from depth images.

        Args:
            depths (list): List of depth images.

        Returns:
            list: Contact pixel information.
        """

        depth = np.concatenate(list(map(self._depth_to_color, depths)), axis=1)
        depth_image = depth.copy()
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
        dp = 1  # Inverse ratio of the accumulator resolution to the image resolution (1 = same resolution)
        minDist = 100  # Minimum distance between the centers of detected circles
        param1 = 50   # Upper threshold for the internal Canny edge detector
        param2 = 10   # Threshold for center detection.
        minRadius = 3  # Minimum radius of the detected circles
        maxRadius = 80  # Maximum radius of the detected circles

        # Apply the Hough Circle Transform
        circles = cv2.HoughCircles(depth_image, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            contact_px = circles[0]
            # Convert the (x, y) coordinates and radius of the circles to integers
            contact_px = np.round(contact_px).astype("int")[0].tolist()
        else:
            contact_px = None
        return contact_px

