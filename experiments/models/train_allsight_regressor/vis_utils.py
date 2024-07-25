import cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

from train_allsight_regressor.geometry import convert_quat_xyzw_to_wxyz
from transformations import quaternion_matrix
from train_allsight_regressor.surface import create_finger_geometry
from scipy import spatial


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

class Display:
    """
    Class for managing and updating a 3D display.

    Attributes:
        statistics (dict): Statistical information for configuring display.
        output_type (str): Type of output for display configuration.
        finger_geometry (tuple): Tuple containing geometric data of the finger.
        tree (scipy.spatial.KDTree): KDTree for nearest neighbor search in finger geometry.
        fig (matplotlib.figure.Figure): Figure object for the plot.
        ax1 (matplotlib.axes._subplots.Axes3DSubplot): Axes object for the 3D plot.
        axbackground (object): Cached background for blitting.
        pred_arrow (matplotlib.lines.Line2D): Line object for predicted arrow in the plot.
        true_arrow (matplotlib.lines.Line2D): Line object for true arrow in the plot.
    """
    def __init__(self, statistics, output_type):
        """
        Initialize the Display object.

        Parameters:
            statistics (dict): Dictionary containing statistical information.
            output_type (str): Type of output for display configuration.
        """
        self.statistics = statistics
        self.output_type = output_type
        self.finger_geometry = create_finger_geometry()
        self.tree = spatial.KDTree(self.finger_geometry[0])

    def config_display(self, blit):
        """
        Configure the display settings for the 3D plot.

        Parameters:
            blit (bool): Whether to use blitting for faster updates.
        """

        plt.close('all')

        self.fig = plt.figure(figsize=(8, 4.4))
        self.ax1 = self.fig.add_subplot(1, 1, 1, projection='3d')

        self.ax1.autoscale(enable=True, axis='both', tight=True)
        self.ax1.set_xlim3d(self.statistics['min'][0], self.statistics['max'][0])
        self.ax1.set_ylim3d(self.statistics['min'][1], self.statistics['max'][1])
        self.ax1.set_zlim3d(self.statistics['min'][2], self.statistics['max'][2])

        self.ax1.tick_params(color='white')
        self.ax1.grid(False)
        self.ax1.set_facecolor('white')
        # First remove fill
        self.ax1.xaxis.pane.fill = False
        self.ax1.yaxis.pane.fill = False
        self.ax1.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        self.ax1.xaxis.pane.set_edgecolor('w')
        self.ax1.yaxis.pane.set_edgecolor('w')
        self.ax1.zaxis.pane.set_edgecolor('w')

        Xc, Yc, Zc = data_for_finger_parametrized(h=0.016, r=0.0128)

        self.ax1.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

        self.ax1.set_yticklabels([])
        self.ax1.set_xticklabels([])
        self.ax1.set_zticklabels([])

        self.pred_arrow, = self.ax1.plot3D([], [], [], color='black', linewidth=5, alpha=0.8)
        self.true_arrow, = self.ax1.plot3D([], [], [], color='red', linewidth=5, alpha=0.8)

        plt.tight_layout()

        self.fig.canvas.draw()
        if blit:
            # cache the background
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax1.bbox)

        plt.show(block=False)

    def update_display(self, y, target=None, blit=True):
        """
        Update the display with new predictions and optionally a target.

        Parameters:
            y (numpy.ndarray): Predicted values.
            target (numpy.ndarray, optional): Target values. Defaults to None.
            blit (bool, optional): Whether to use blitting for faster updates. Defaults to True.
        """
        scale = 1500
        pred_pose = y[:3]
        pred_force = y[3:6]
        depth = round((5e-3 - y[-1]) * 1000, 2)
        torque = round(y[-2], 4)

        if 'torque' in self.output_type and 'depth' in self.output_type:
            self.fig.suptitle(f'\nForce: {y[3:6]} (N)'
                              f'\nPose: {y[:3] * 1000} (mm)'
                              f'\nTorsion: {torque} (Nm)'
                              f'\nDepth: {abs(depth)} (mm)',
                              fontsize=13)

        if target is not None:
            true_pose = target[:3]
            true_force = target[3:6]
            _, ind = self.tree.query(true_pose)
            cur_rot = self.finger_geometry[1][ind].copy()
            true_rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(cur_rot))
            true_force_transformed = np.dot(true_rot[:3, :3], true_force)

            self.true_arrow.set_xdata(np.array([true_pose[0], true_pose[0] + true_force_transformed[0] / scale]))
            self.true_arrow.set_ydata(np.array([true_pose[1], true_pose[1] + true_force_transformed[1] / scale]))
            self.true_arrow.set_3d_properties(
                np.array([true_pose[2], true_pose[2] + true_force_transformed[2] / scale]))

        _, ind = self.tree.query(pred_pose)
        cur_rot = self.finger_geometry[1][ind].copy()
        pred_rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(cur_rot))
        pred_force_transformed = np.dot(pred_rot[:3, :3], pred_force)

        self.pred_arrow.set_xdata(np.array([pred_pose[0], pred_pose[0] + pred_force_transformed[0] / scale]))
        self.pred_arrow.set_ydata(np.array([pred_pose[1], pred_pose[1] + pred_force_transformed[1] / scale]))
        self.pred_arrow.set_3d_properties(np.array([pred_pose[2], pred_pose[2] + pred_force_transformed[2] / scale]))

        if blit:
            self.fig.canvas.restore_region(self.axbackground)
            self.ax1.draw_artist(self.pred_arrow)
            self.ax1.draw_artist(self.true_arrow)
            self.fig.canvas.blit(self.ax1.bbox)
        else:
            self.fig.canvas.draw()

        self.fig.canvas.flush_events()


class Arrow3D(FancyArrowPatch):
    """
    Class for representing a 3D arrow.

    Inherits from FancyArrowPatch.

    Attributes:
        _verts3d (tuple): Tuple of X, Y, Z coordinates for the arrow.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        Initialize a 3D arrow.

        Parameters:
            xs (float): X coordinates.
            ys (float): Y coordinates.
            zs (float): Z coordinates.
        """
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """
        Perform 3D projection for the arrow.

        Parameters:
            renderer: Renderer object. Defaults to None.
        
        Returns:
            float: Minimum value of the Z coordinates.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def update_line(hl, new_data):
    """
    Update a 3D line object with new data.

    Parameters:
        hl (matplotlib.lines.Line3D): Line object to update.
        new_data (list): List of new data points for X, Y, Z coordinates.
    """
    hl.set_xdata(np.asarray(new_data[0]))
    hl.set_ydata(np.asarray(new_data[1]))
    hl.set_3d_properties(np.asarray(new_data[2]))


class Annotation3D(Annotation):
    """
    Class for annotating a point in 3D space with text.

    Inherits from Annotation.

    Attributes:
        _verts3d (tuple): Tuple of X, Y, Z coordinates for the annotation.
    """

    def __init__(self, s, xyz, *args, **kwargs):
        """
        Initialize the 3D annotation.

        Parameters:
            s (str): Text to annotate.
            xyz (tuple): Tuple of coordinates (x, y, z).
        """
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        """
        Draw the 3D annotation on the plot.

        Parameters:
            renderer: Renderer object.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)


def update_arrow(hl, new_data):
    """
    Update a 3D arrow object with new data.

    Parameters:
        hl (matplotlib.lines.Line2D): Arrow object to update.
        new_data (list): List of new data points for X, Y, Z coordinates.
    """
    hl.set_xdata(np.asarray([new_data[0][0], new_data[0][1]]))
    hl.set_ydata(np.asarray([new_data[1][0], new_data[1][1]]))
    hl.set_3d_properties(np.asarray([new_data[2][0], new_data[2][1]]))


def annotate3D(ax, s, *args, **kwargs):
    """
    Add a 3D annotation to a Axes3D object.

    Parameters:
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Axes object to add annotation to.
        s (str): Text for annotation.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)


def data_for_finger_parametrized(h=0.016 * 1000, r=0.0125 * 1000):
    """
    Generate parametrized data for a finger geometry.

    Parameters:
        h (float): Height parameter. Defaults to 16 mm.
        r (float): Radius parameter. Defaults to 12.5 mm.

    Returns:
        tuple: Tuple containing X, Y, Z grids for the finger geometry.
    """
    H = h + r

    def radius(z):
        if z < h:
            return r
        else:
            return np.sqrt(r ** 2 - (z - h) ** 2)

    def radius_dz(z):
        if z < h:
            return 0
        else:
            return h - z

    # 100
    z = np.linspace(0, H, 30)
    q = np.linspace(0, 2 * np.pi, 20)

    f = np.where(z < h, r,
                 np.sqrt(r ** 2 - (z - h) ** 2)
                 )

    z_grid, Q = np.meshgrid(z, q)

    x_grid = f * np.cos(Q)
    y_grid = f * np.sin(Q)

    return x_grid, y_grid, z_grid


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    """
    Generate data points for a cylinder along the Z-axis.

    Parameters:
        center_x (float): X coordinate of cylinder center.
        center_y (float): Y coordinate of cylinder center.
        radius (float): Radius of the cylinder.
        height_z (float): Height of the cylinder along the Z-axis.

    Returns:
        tuple: Tuple containing X, Y, Z grids for the cylinder.
    """
    z = np.linspace(0, height_z, 2)
    theta = np.linspace(0, 2 * np.pi, 15)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def data_for_sphere_along_z(center_x, center_y, radius, height_z):
    """
    Generate data points for a sphere along the Z-axis.

    Parameters:
        center_x (float): X coordinate of sphere center.
        center_y (float): Y coordinate of sphere center.
        radius (float): Radius of the sphere.
        height_z (float): Height of the sphere along the Z-axis.

    Returns:
        tuple: Tuple containing X, Y, Z grids for the sphere.
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
    Set equal scale for axes of a 3D plot.

    Parameters:
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Axes object of the 3D plot.
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
    Class for selecting points using mouse interactions on an image.

    Attributes:
        windowname (str): Name of the window for displaying the image.
        img1 (numpy.ndarray): Original image array.
        img (numpy.ndarray): Modified image array.
        curr_pt (list): Current point coordinates.
        point (list): List of selected points.
        r (int): Radius of the selection circle.
    """
    def __init__(self, windowname, img, rad):
        """
        Initialize the MousePts object.

        Parameters:
            windowname (str): Name of the window for displaying the image.
            img (numpy.ndarray): Image array to display.
            rad (int): Radius of the selection circle.
        """

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
        Handle mouse events for selecting points.

        Parameters:
            event: Event type.
            x (int): X coordinate of the mouse event.
            y (int): Y coordinate of the mouse event.
            flags: Flags for the mouse event.
            param: Additional parameters.
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
        Get selected points using mouse interactions.

        Parameters:
            count (int, optional): Number of points to select. Defaults to 1.
            img (numpy.ndarray, optional): Image array to display. Defaults to None.

        Returns:
            tuple: Tuple containing selected points and modified image array.
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
