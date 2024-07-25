import cv2
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs
#
#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         FancyArrowPatch.draw(self, renderer)

class Arrow3D(FancyArrowPatch):
    """
    Extended from `FancyArrowPatch` to represent an arrow in 3D space.

    Args:
        xs (float): X-coordinate of the arrow's start point.
        ys (float): Y-coordinate of the arrow's start point.
        zs (float): Z-coordinate of the arrow's start point.
        *args: Additional positional arguments passed to `FancyArrowPatch`.
        **kwargs: Additional keyword arguments passed to `FancyArrowPatch`.

    Methods:
        do_3d_projection(self, renderer=None):
            Project the arrow onto a 3D renderer and return the minimum Z value.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """
        Project the arrow onto a 3D renderer and return the minimum Z value.

        Args:
            renderer (RendererBase, optional): The 3D renderer to use for projection.

        Returns:
            float: Minimum Z value of the projected arrow.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def update_line(hl, new_data):
    """
    Update the data of a 3D line plot.

    Args:
        hl (Line3D): The Line3D object to be updated.
        new_data (tuple): Tuple containing new X, Y, Z data arrays.

    Returns:
        None
    """
    hl.set_xdata(np.asarray(new_data[0]))
    hl.set_ydata(np.asarray(new_data[1]))
    hl.set_3d_properties(np.asarray(new_data[2]))


def update_arrow(hl, new_data):
    """
    Update the data of a 3D arrow plot.

    Args:
        hl (Arrow3D): The Arrow3D object to be updated.
        new_data (tuple): Tuple containing new X, Y, Z data arrays for arrow points.

    Returns:
        None
    """
    hl.set_xdata(np.asarray([new_data[0][0], new_data[0][1]]))
    hl.set_ydata(np.asarray([new_data[1][0], new_data[1][1]]))
    hl.set_3d_properties(np.asarray([new_data[2][0], new_data[2][1]]))


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    """
    Generate data points for a cylinder aligned along the Z-axis.

    Args:
        center_x (float): X-coordinate of the cylinder's center.
        center_y (float): Y-coordinate of the cylinder's center.
        radius (float): Radius of the cylinder.
        height_z (float): Height of the cylinder along the Z-axis.

    Returns:
        tuple: Tuple containing X, Y, Z grids representing the cylinder.
    """
    z = np.linspace(0, height_z, 2)
    theta = np.linspace(0, 2 * np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def data_for_sphere_along_z(center_x, center_y, radius, height_z):
    """
    Generate data points for a sphere aligned along the Z-axis.

    Args:
        center_x (float): X-coordinate of the sphere's center.
        center_y (float): Y-coordinate of the sphere's center.
        radius (float): Radius of the sphere.
        height_z (float): Height of the sphere along the Z-axis.

    Returns:
        tuple: Tuple containing X, Y, Z grids representing the sphere.
    """
    q = np.linspace(0, 2 * np.pi, 15)
    p = np.linspace(0, np.pi / 2, 15)
    p_, q_ = np.meshgrid(q, p)
    x_grid = radius * np.cos(p_) * np.sin(q_)
    y_grid = radius * np.sin(p_) * np.sin(q_)
    z_grid = radius * np.cos(q_) + height_z
    return x_grid, y_grid, z_grid


def set_axes_equal(ax):
    """
    Set equal scale for 3D plot axes to maintain proportions of objects.

    Args:
        ax (Axes3D): The 3D axes object to set equal scale.

    Returns:
        None
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class MousePts:
    """
    Class to handle mouse events for selecting points on an image.

    Args:
        windowname (str): Name of the OpenCV window.
        img (numpy.ndarray): Input image on which points are selected.
        rad (int): Radius for the circle to be drawn around selected points.

    Methods:
        select_point(self, event, x, y, flags, param):
            Callback function for handling mouse events.
        getpt(self, count=1, img=None):
            Get user-selected points from the image.

    Attributes:
        windowname (str): Name of the OpenCV window.
        img1 (numpy.ndarray): Copy of the original input image.
        img (numpy.ndarray): Image on which points are selected.
        curr_pt (list): Current point selected by the user.
        point (list): List of points selected by the user.
        r (int): Radius of the circle drawn around selected points.
    """
    def __init__(self, windowname, img, rad):

        self.windowname = windowname
        self.img1 = img.copy()
        self.img = self.img1.copy()
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(windowname, img)
        self.curr_pt = []
        self.point = []
        self.r = max(min(rad, 50), 5)

    def select_point(self, event, x, y, flags, param):
        """
        Callback function for handling mouse events.

        Args:
            event (int): Type of mouse event (e.g., left button down, mouse move).
            x (int): X-coordinate of the mouse cursor position.
            y (int): Y-coordinate of the mouse cursor position.
            flags (int): Additional flags passed by OpenCV.
            param (object): Additional parameters passed by OpenCV.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x, y])
            self.img = cv2.circle(self.img, (x, y), self.r, (0, 255, 0), 2)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.curr_pt = [x, y]
            # print(self.point)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.r -= 5
            # cv2.circle(self.img, (x, y), 50, (0, 255, 0), -1)

    def getpt(self, count=1, img=None):
        """
        Get user-selected points from the image.

        Args:
            count (int, optional): Number of points to select. Defaults to 1.
            img (numpy.ndarray, optional): Image on which to select points. Defaults to None.

        Returns:
            tuple: Tuple containing list of selected points and the image with points drawn.
        """
        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.img)
        cv2.setMouseCallback(self.windowname, self.select_point)
        self.point = []

        while True:
            cv2.imshow(self.windowname, self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.point) >= count:
                break
            # print(self.point)

        cv2.setMouseCallback(self.windowname, lambda *args: None)
        # cv2.destroyAllWindows()
        self.point.append(self.r)
        return self.point, self.img