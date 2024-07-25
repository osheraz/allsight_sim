import numpy as np
import cv2
import math
import torch

FIX = 0


def concatenate_matrices(*matrices):
    """
    Concatenates multiple 4x4 transformation matrices into a single matrix.

    Args:
        *matrices: Variable length argument list of 4x4 numpy arrays.

    Returns:
        np.ndarray: Concatenated 4x4 transformation matrix.
    """

    M = np.identity(4)
    for i in matrices:
        M = np.dot(M, i)
    return M


def unit_vector(data, axis=None, out=None):
    """
    Returns the unit vector of a numpy array along a specified axis.

    Args:
        data (np.ndarray): Input array.
        axis (int, optional): Axis along which to compute the unit vector. If None, the entire array is considered.
        out (np.ndarray, optional): Output array to store the result.

    Returns:
        np.ndarray: Unit vector of the input array.
    """

    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """
    Returns a 4x4 rotation matrix based on a specified angle and direction vector.

    Args:
        angle (float): Angle of rotation in radians.
        direction (np.ndarray): Direction vector (3D) around which to rotate.
        point (np.ndarray, optional): Point to rotate around. If None, rotation is around origin.

    Returns:
        np.ndarray: 4x4 rotation matrix.
    """

    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def convert_quat_xyzw_to_wxyz(q):
    """
    Converts a quaternion from XYZW to WXYZ format.

    Args:
        q (list or np.ndarray): Input quaternion in XYZW format.

    Returns:
        list: Quaternion converted to WXYZ format.
    """
    q=list(q)
    q[0], q[1], q[2], q[3] = q[3], q[0], q[1], q[2]
    return q

def convert_quat_wxyz_to_xyzw(q):
    """
    Converts a quaternion from WXYZ to XYZW format.

    Args:
        q (list or np.ndarray): Input quaternion in WXYZ format.

    Returns:
        list: Quaternion converted to XYZW format.
    """
    q=list(q)
    q[3], q[0], q[1], q[2] = q[0], q[1], q[2], q[3]
    return q

def T_inv(T_in):
    """
    Computes the inverse of a 4x4 transformation matrix.

    Args:
        T_in (np.ndarray): Input 4x4 transformation matrix.

    Returns:
        np.ndarray: Inverse of the input matrix.
    """
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))



def crop_image(img, pad):
    """
    Crops an image by removing padding from its borders.

    Args:
        img (np.ndarray): Input image.
        pad (int): Number of pixels to pad on each side.

    Returns:
        np.ndarray: Cropped image.
    """
    return img[pad:-pad, pad:-pad]

def _diff(target, base):
    """
    Computes the absolute difference between two images.

    Args:
        target (np.ndarray): Target image.
        base (np.ndarray): Base image.

    Returns:
        np.ndarray: Absolute difference between target and base images.
    """
    diff = (target * 1.0 - base) / 255.0 + 0.5
    diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 1.0 + 0.5
    diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
    return diff_abs

class ContactArea:
    """
    Computes and optionally draws the contact area between two images.
    """
    def __init__(
        self, base=None, draw_poly=False, contour_threshold=100,real_time=True,*args, **kwargs
    ):
        """
        Initializes the ContactArea instance.

        Args:
            base (np.ndarray, optional): Base image for comparison.
            draw_poly (bool, optional): Whether to draw the contact area polygon.
            contour_threshold (int, optional): Minimum contour length to consider as a contact area.
            real_time (bool, optional): Whether to compute contact area in real-time.
        """
        self.base = base
        self.draw_poly = draw_poly
        self.contour_threshold = contour_threshold
        self.real_time = real_time

    def __call__(self, target, base=None):
        """
        Computes the contact area between the target and base images.

        Args:
            target (np.ndarray): Target image.
            base (np.ndarray, optional): Base image for comparison.

        Returns:
            tuple: Contact area information including polygon points and axes.
        """
        base = self.base if base is None else base
        if base is None:
            raise AssertionError("A base sample must be specified for Pose.")
        diff = self._diff(target, base)
        diff = self._smooth(diff)
        contours = self._contours(diff)
        if self._compute_contact_area(contours, self.contour_threshold) == None and self.real_time==False:
            raise Exception("No contact area detected.")
        if self._compute_contact_area(contours, self.contour_threshold) == None and self.real_time==True:
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
        Computes the difference map between the target and base images.

        Args:
            target (np.ndarray): Target image.
            base (np.ndarray): Base image.

        Returns:
            np.ndarray: Difference map between target and base images.
        """
        diff = (target * 1.0 - base) / 255.0 + 0.5
        diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
        diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
        return diff_abs

    def _smooth(self, target):
        """
        Applies smoothing to the input image using a kernel.

        Args:
            target (np.ndarray): Input image.

        Returns:
            np.ndarray: Smoothed image.
        """
        kernel = np.ones((64, 64), np.float32)
        kernel /= kernel.sum()
        diff_blur = cv2.filter2D(target, -1, kernel)
        return diff_blur

    def _contours(self, target):
        """
        Finds contours in the input image using a threshold.

        Args:
            target (np.ndarray): Input image.

        Returns:
            list: List of contours found in the image.
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
        Draws major and minor axes on the input image.

        Args:
            target (np.ndarray): Input image.
            poly (np.ndarray): Polygon points of the contact area.
            major_axis (np.ndarray): Start point of the major axis.
            major_axis_end (np.ndarray): End point of the major axis.
            minor_axis (np.ndarray): Start point of the minor axis.
            minor_axis_end (np.ndarray): End point of the minor axis.
            lineThickness (int, optional): Thickness of lines to draw.
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
        """
        Computes the contact area from detected contours.

        Args:
            contours (list): List of contours.
            contour_threshold (int): Minimum length of contour to be considered as a contact area.

        Returns:
            tuple: Contact area information including polygon points and axes.
        """
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


def resizeAndPad(img, size, padColor=255):
    """
    Resizes an image while maintaining its aspect ratio and padding if necessary.

    Args:
        img (np.ndarray): Input image.
        size (tuple): Desired output size (width, height).
        padColor (int or tuple, optional): Color value for padding. Defaults to 255.

    Returns:
        np.ndarray: Resized and padded image.
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


def circle_mask(size=(640, 480), border=0):
    """
    Generates a circular mask image of a specified size.

    Args:
        size (tuple, optional): Size of the mask image (width, height). Defaults to (640, 480).
        border (int, optional): Border thickness. Defaults to 0.

    Returns:
        np.ndarray: Circular mask image.
    """
    m = np.zeros((size[1], size[0]))
    m_center = (size[0] // 2 - FIX, size[1] // 2)
    m_radius = min(size[0], size[1]) // 2 - border
    m = cv2.circle(m, m_center, m_radius, 255, -1)
    m /= 255
    m = m.astype(np.float32)
    mask = np.stack([m, m, m], axis=2)
    return mask


def get_coords(x, y, angle, imwidth, imheight):
    """
    Computes the coordinates of two points based on an angle and image dimensions.

    Args:
        x (int): X-coordinate of the starting point.
        y (int): Y-coordinate of the starting point.
        angle (float): Angle in radians.
        imwidth (int): Image width.
        imheight (int): Image height.

    Returns:
        tuple: Coordinates of two points ((endx1, endy1), (endx2, endy2)).
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
    Interpolates an image tensor to a specified size.

    Args:
        img (torch.Tensor): Input image tensor of shape (C x H x W).
        rows (int): Desired number of rows.
        cols (int): Desired number of columns.

    Returns:
        torch.Tensor: Interpolated image tensor of shape (C x rows x cols).
    """

    img = torch.nn.functional.interpolate(img, size=cols)
    img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=rows)
    img = img.permute(0, 2, 1)

    return img

if __name__ == "__main__":

    import cv2 as cv
    import numpy as np

    # The video feed is read in as
    # a VideoCapture object
    cap = cv.VideoCapture(4)
    c_mask = circle_mask((640, 480))
    c_mask = np.stack([c_mask, c_mask, c_mask], axis=2)
    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    ret, first_frame = cap.read()

    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    while (cap.isOpened()):

        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        ret, frame = cap.read()


        frame = frame * c_mask
        # Opens a new window and displays the input
        # frame
        cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", rgb)

        # Updates previous frame
        prev_gray = gray

        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and
    # closes all windows
    cap.release()
    cv.destroyAllWindows()