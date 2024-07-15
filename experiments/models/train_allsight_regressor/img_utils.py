import numpy as np
import cv2
import math
import torch


def crop_image(img, pad):
    """
    Crops the image by removing padding from all sides.

    Parameters:
    - img (numpy.ndarray): Input image to be cropped.
    - pad (int): Number of pixels to crop from each side.

    Returns:
    - numpy.ndarray: Cropped image.
    """
    return img[pad:-pad, pad:-pad]


def resize_and_pad(img, size, padColor=255):
    """
    Resizes the input image to the specified size and pads if necessary.

    Parameters:
    - img (numpy.ndarray): Input image to be resized.
    - size (tuple): Desired output size (height, width).
    - padColor (int or tuple): Color for the padding (default is 255, white).

    Returns:
    - numpy.ndarray: Resized and padded image.
    """

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA

    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w) / h
    saspect = float(sw) / sh

    if (saspect >= aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img


def square_cut(img, size=480):
    """
    Cuts the input image to a square shape of a specified size and pads if necessary.

    Parameters:
    - img (numpy.ndarray): Input image to be processed.
    - size (int): Desired size of the square image (default is 480).

    Returns:
    - numpy.ndarray: Square-shaped image with optional padding.
    """

    col_sum = np.where(np.sum(img, axis=0) > 0)
    row_sum = np.where(np.sum(img, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    img = img[y1:y2, x1:x2 - 1]

    zero_axis_fill = max(0, (size - img.shape[0]))
    one_axis_fill = max((size - img.shape[1]), 0)

    top = zero_axis_fill // 2
    bottom = zero_axis_fill - top
    left = one_axis_fill // 2
    right = one_axis_fill - left

    padded_img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant')

    # print(padded_img.shape)
    return padded_img


def circle_mask(size=(640, 480), border=0):
    """
    Generates a circular mask image.

    Parameters:
    - size (tuple): Size of the output mask image (height, width).
    - border (int): Border thickness around the circle (default is 0).

    Returns:
    - numpy.ndarray: Circular mask image with values normalized to [0, 1].
    """
    m = np.zeros((size[1], size[0]))
    m_center = (size[0] // 2, size[1] // 2)
    m_radius = min(size[0], size[1]) // 2 - border
    m = cv2.circle(m, m_center, m_radius, 255, -1)
    m /= 255
    m = m.astype(np.float32)
    mask = np.stack([m, m, m], axis=2)
    return mask


def get_coords(x, y, angle, imwidth, imheight):
    """
    Calculates coordinates of two points based on a given point, angle, and image dimensions.

    Parameters:
    - x (float): x-coordinate of the starting point.
    - y (float): y-coordinate of the starting point.
    - angle (float): Angle in radians for the direction.
    - imwidth (int): Width of the image.
    - imheight (int): Height of the image.

    Returns:
    - tuple: Coordinates of two points ((endx1, endy1), (endx2, endy2)).
    """
    x1_length = (imwidth - x) / (math.cos(angle) + 1e-6)
    y1_length = (imheight - y) / (math.sin(angle) + 1e-6)
    length = max(abs(x1_length), abs(y1_length))
    endx1 = x + length * math.cos(angle)
    endy1 = y + length * math.sin(angle)

    x2_length = (imwidth - x) / (math.cos(angle + math.pi) + 1e-6)
    y2_length = (imheight - y) / (math.sin(angle + math.pi) + 1e-6)
    length = max(abs(x2_length), abs(y2_length))
    endx2 = x + length * math.cos((angle + math.pi))
    endy2 = y + length * math.sin((angle + math.pi))

    return (int(endx1), int(endy1)), (int(endx2), int(endy2))


def interpolate_img(img, rows, cols):
    """
    Interpolates an image tensor to a specified size using bilinear interpolation.

    Parameters:
    - img (torch.Tensor): Input image tensor of shape (channels, height, width).
    - rows (int): Desired number of rows (height) for the output image.
    - cols (int): Desired number of columns (width) for the output image.

    Returns:
    - torch.Tensor: Interpolated image tensor of shape (channels, rows, cols).
    """

    img = torch.nn.functional.interpolate(img, size=cols)
    img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=rows)
    img = img.permute(0, 2, 1)

    return img


# def _diff(target, base):
#     diff = (target * 1.0 - base) / 255.0 + 0.5
#     diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 1.0 + 0.5
#     diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
#     return diff_abs

def _structure(target, size=None):
    """
    Performs structure analysis on the input image.

    Parameters:
    - target (numpy.ndarray): Input image to be analyzed.
    - size (tuple or None): Optional size of the output image (height, width).

    Returns:
    - numpy.ndarray: Processed image after structure analysis.
    """
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    gray_circle = cv2.adaptiveThreshold(
        gray_target, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -30)

    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 0.2
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.minThreshold = 1
    params.thresholdStep = 1
    params.maxThreshold = 255
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(target)

    # im_with_keypoints = cv2.drawKeypoints(gray_circle, keypoints, np.array([]), (255, 255, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img = gray_circle.copy()
    for x in range(1, len(keypoints)):
        img = cv2.circle(img, (int(keypoints[x].pt[0]), int(keypoints[x].pt[1])),
                         radius=int(min(keypoints[x].size, 5)), color=(255,255,255), thickness=-1)

    return img if size is None else cv2.resize(img, size)

def _diff(target, base):
    """
    Computes the absolute difference between two images.

    Parameters:
    - target (numpy.ndarray): Target image.
    - base (numpy.ndarray): Base image.

    Returns:
    - numpy.ndarray: Absolute difference between target and base images.
    """

    diff_raw = target - base
    diff_mask = (diff_raw < 150) # .astype(np.uint8)
    diff = diff_raw * diff_mask
    return diff

def isInside(x, y, size=(640,480), border=0):
    """
    Checks if a point (x, y) lies inside a circular area.

    Parameters:
    - x (int): x-coordinate of the point.
    - y (int): y-coordinate of the point.
    - size (tuple): Size of the circular area (height, width).
    - border (int): Border thickness around the circle (default is 0).

    Returns:
    - bool: True if the point is inside the circle, False otherwise.
    """

    m_center = (size[0] // 2, size[1] // 2)
    m_radius = min(size[0], size[1]) // 2 - border

    circle_x, circle_y = m_center
    if ((x - circle_x) * (x - circle_x) +
            (y - circle_y) * (y - circle_y) <= m_radius * m_radius):
        return True
    else:
        return False


def _diff_abs(target, base):
    """
    Computes the absolute difference between two images and returns the mean absolute difference.

    Parameters:
    - target (numpy.ndarray): Target image.
    - base (numpy.ndarray): Base image.

    Returns:
    - numpy.ndarray: Mean absolute difference between target and base images.
    """

    diff = (target * 1.0 - base) / 255.0 + 0.5
    diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 1.0 + 0.5
    diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)

    return diff_abs


class ContactArea:
    """
    Class to compute and visualize contact area between two images.

    Initialization Parameters:
    - base (numpy.ndarray or None): Reference image for comparison.
    - draw_poly (bool): Flag to enable/disable drawing polygons on the images (default is False).
    - contour_threshold (int): Threshold for contour length to consider (default is 100).
    - real_time (bool): Flag to enable real-time processing (default is True).

    Methods:
    - __call__(target, base=None): Computes the contact area between the target image and the base image.
    Returns the polygon coordinates and major/minor axes coordinates if found.

    Attributes:
    - base (numpy.ndarray or None): Reference image for comparison.
    - draw_poly (bool): Flag to enable/disable drawing polygons on the images.
    - contour_threshold (int): Threshold for contour length to consider.
    - real_time (bool): Flag to enable real-time processing.
    """
    def __init__(
            self, base=None, draw_poly=False, contour_threshold=100, real_time=True, *args, **kwargs
    ):
        self.base = base
        self.draw_poly = draw_poly
        self.contour_threshold = contour_threshold
        self.real_time = real_time

    def __call__(self, target, base=None):
        base = self.base if base is None else base
        if base is None:
            raise AssertionError("A base sample must be specified for Pose.")
        diff = self._diff(target, base)
        diff = self._smooth(diff)
        contours = self._contours(diff)
        if self._compute_contact_area(contours, self.contour_threshold) == None and self.real_time == False:
            raise Exception("No contact area detected.")
        if self._compute_contact_area(contours, self.contour_threshold) == None and self.real_time == True:
            return None
        else:
            (
                poly,
                major_axis,
                major_axis_end,
                minor_axis,
                minor_axis_end,
            ) = self._compute_contact_area(contours, self.contour_threshold)
        if self.draw_poly:
            self._draw_major_minor(
                target, poly, major_axis, major_axis_end, minor_axis, minor_axis_end
            )
        return poly, major_axis, major_axis_end, minor_axis, minor_axis_end

    def _diff(self, target, base):
        """
        Computes the normalized difference between the target and base images.

        Parameters:
        - target (numpy.ndarray): Target image.
        - base (numpy.ndarray): Base image.

        Returns:
        - numpy.ndarray: Normalized difference between target and base images.
        """

        diff = (target * 1.0 - base) / 255.0 + 0.5
        diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
        diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
        return diff_abs

    def _smooth(self, target):
        """
        Applies smoothing to the target image using a predefined kernel.

        Parameters:
        - target (numpy.ndarray): Input image.

        Returns:
        - numpy.ndarray: Smoothed image.
        """
        kernel = np.ones((64, 64), np.float32)
        kernel /= kernel.sum()
        diff_blur = cv2.filter2D(target, -1, kernel)
        return diff_blur

    def _contours(self, target):
        """
        Finds contours in the target image after thresholding and erosion.

        Parameters:
        - target (numpy.ndarray): Input image.

        Returns:
        - list: List of contours found in the image.
        """
        mask = ((np.abs(target) > 0.04) * 255).astype(np.uint8)
        kernel = np.ones((16, 16), np.uint8)
        mask = cv2.erode(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _draw_major_minor(
            self,
            target,
            poly,
            major_axis,
            major_axis_end,
            minor_axis,
            minor_axis_end,
            lineThickness=2,
    ):
        """
        Draws the major and minor axes and the polygon on the target image.

        Parameters:
        - target (numpy.ndarray): Image to draw on.
        - poly (numpy.ndarray): Polygon points.
        - major_axis (numpy.ndarray): Starting point of the major axis.
        - major_axis_end (numpy.ndarray): End point of the major axis.
        - minor_axis (numpy.ndarray): Starting point of the minor axis.
        - minor_axis_end (numpy.ndarray): End point of the minor axis.
        - lineThickness (int, optional): Thickness of lines. Default is 2.
        """
        cv2.polylines(target, [poly], True, (255, 255, 255), lineThickness)
        cv2.line(
            target,
            (int(major_axis_end[0]), int(major_axis_end[1])),
            (int(major_axis[0]), int(major_axis[1])),
            (0, 0, 255),
            lineThickness,
        )
        cv2.line(
            target,
            (int(minor_axis_end[0]), int(minor_axis_end[1])),
            (int(minor_axis[0]), int(minor_axis[1])),
            (0, 255, 0),
            lineThickness,
        )

    def _compute_contact_area(self, contours, contour_threshold):
        for contour in contours:
            if len(contour) > contour_threshold:
                ellipse = cv2.fitEllipse(contour)
                poly = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                    int(ellipse[2]),
                    0,
                    360,
                    5,
                )
                center = np.array([ellipse[0][0], ellipse[0][1]])
                a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2)
                theta = (ellipse[2] / 180.0) * np.pi
                major_axis = np.array(
                    [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
                )
                minor_axis = np.array(
                    [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
                )
                major_axis_end = 2 * center - major_axis
                minor_axis_end = 2 * center - minor_axis
                return poly, major_axis, major_axis_end, minor_axis, minor_axis_end
