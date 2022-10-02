
"""
This file will contain all the different funcitons and classes which can be used to interpret the
different cxr from Neural Nets

# ? Maybes
 - Cobbs angle
"""
__all__ = ['LungInterpretation']

from unicodedata import numeric
import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from cv2 import RETR_EXTERNAL
from pkg_resources import ExtractionError
from scipy.interpolate import splev, splprep
from tensorflow import keras


class LungIntepretation:
    """
    This class will be used to take out different metrics from a given lung image or the
    contours if nothing else is provided

    Metrics Available
    ---
     - lung_vertical_span_ratio
     - area_ratio
     - height_width_ratio
     - ct_ratio
     - cp_angle
     - cp_angle_relative_position
    """

    #* Some HyperParameters for our model
    # For smoothening the xray contours
    SMOOTHENING = 400
    RIGHT_LUNG_MEDIASTINAL_POINT_WEIGHTS = (1.4, 0.7)
    LEFT_LUNG_MEDIASTINAL_POINT_WEIGHTS = (-1.195, 0.95)
    CP_ANGLE_RANGE = range(1, 6)

    

    def __init__(self, mask) -> None:
        self.ll = {"side": "left"}
        self.rl = {"side": "right"}

        self._genereate_contours(mask)
        self._extract_lung_dimensions()

    @property
    def contours(self):
        return self.rl["contours"], self.ll["contours"]

    @property
    def contours_polar(self):
        right = list(zip(self.rl["contours"][0], self.rl["contours"][1]))
        left = list(zip(self.ll["contours"][0], self.ll["contours"][1]))
        return right, left

    @property
    def lung_vertical_span_ratio(self) -> float:
        """
        Calculates the ratio of the vertical lung span of the right vs left.

        This is because usually the right lung is a bit above, and we wanted
        to keep the ratio below 1.
        """
        return round(self.rl["span"] / self.ll["span"], 3)

    @property
    def area_ratio(self):
        """
        Caculates Area ratio, right lung / left lung
        """
        ll_area = self.ll["area"]
        rl_area = self.rl["area"]
        return rl_area / ll_area

    @property
    def height_width_ratio(self):
        width = self.max_thoracic_width()[1]
        return self.bigger_lung["span"] / width

    def ct_ratio(self, plot=False):
        """
        The cardiothoracic ratio is measured on a PA chest x-ray, and is the ratio of maximal horizontal cardiac diameter to maximal horizontal thoracic diameter (inner edge of ribs/edge of pleura).

        A normal measurement is 0.42-0.50. A measurement <0.42 is usually deemed to be pathologic. A measurement >0.50 is usually taken to be abnormal although some radiologists feel that measurements up to 0.55 are "borderline".

        Returns
        ---

        A tuple with

            1. The Original CT Ratio
            2. The Cardio - Pulmonary Area, ratio of areas of mediastinum with area of lung

        """
        rl, ll = self.contours_polar

        # Using different weights, finding the best mediastinal points which can be useful
        rl_mediastinal_points = sorted(
            rl,
            key=lambda x: self.RIGHT_LUNG_MEDIASTINAL_POINT_WEIGHTS[0] * x[0]
            + self.RIGHT_LUNG_MEDIASTINAL_POINT_WEIGHTS[1] * x[1],
            reverse=True,
        )[:3]

        ll_mediastinal_points = sorted(
            ll,
            key=lambda x: self.LEFT_LUNG_MEDIASTINAL_POINT_WEIGHTS[0] * x[0]
            + self.LEFT_LUNG_MEDIASTINAL_POINT_WEIGHTS[1] * x[1],
            reverse=True,
        )[:3]

        # Out of those selected, getting the largest point, so that we dont miss anything
        search_bottom_right = sorted(rl_mediastinal_points, key=lambda x: x[1], reverse=True)[0]

        threshold_img = self.threshold_img
        left_cp_angle_point = self._left_cp_angle_coordinates
        search_top_right = self.rl["left"]
        search_top_left = self.ll["right"]
        thresh_copy = threshold_img.copy()

        idx_top_right = rl.index(search_top_right)
        idx_bottom_right = rl.index(search_bottom_right)
        idx_top_left = ll.index(search_top_left)
        idx_bottom_left = ll.index(left_cp_angle_point)

        right_useful_slice = rl[idx_bottom_right : idx_top_right + 1]
        left_useful_slice = ll[idx_top_left : idx_bottom_left + 1]
        thresh_copy = threshold_img.copy()

        # * Looping through the image, and drawing every pixel below the lung
        for i, point in enumerate(left_useful_slice):
            try:
                curr_point = point
                next_point = left_useful_slice[i + 1]
                thresh_copy[curr_point[1] :, curr_point[0] : next_point[0]] = 1
            except IndexError as error:
                pass

        for i, point in enumerate(right_useful_slice):
            try:
                curr_point = point
                next_point = left_useful_slice[i + 1]
                thresh_copy[curr_point[1] :, curr_point[0] : next_point[0]] = 1
            except IndexError as error:
                pass

        # Summing the pixels of the image row wise, and subtracting them
        original_col_sum = np.sum(threshold_img, 1)
        total_col_sum = np.sum(thresh_copy, 1)

        # This array has the subtracted values, ie mediastinal values
        subtracted_array = np.subtract(total_col_sum, original_col_sum)

        # Filtering out the relevant amount of info from the whole 255 row array
        mediastinal_array = []
        for i in range(256):
            if i < min(search_top_right[1], search_top_left[1]) or i > search_bottom_right[1]:
                continue
            mediastinal_array.append((i, subtracted_array[i]))

        total_mediastinum_area = np.sum(mediastinal_array, 0)
        total_lung_area = self.ll["area"] + self.rl["area"]
        self._ct_area = total_mediastinum_area[1] / total_lung_area

        max_mediastinal_widths = sorted(mediastinal_array, key=lambda x: x[1], reverse=True)[:5]
        thoracic_width = self.max_thoracic_width()
        mean_max_mediastinal_widths = np.mean(max_mediastinal_widths, axis=0)

        if plot:
            plt.plot(
                (0, 255),
                (int(mean_max_mediastinal_widths[0]), int(mean_max_mediastinal_widths[0])),
                label="Max Mediastinal Width",
            )
            plt.plot((0, 255), (thoracic_width[0], thoracic_width[0]), label="Max Thoracic Width")
            plt.legend()

        return round(mean_max_mediastinal_widths[1] / thoracic_width[1], 3), np.round(self._ct_area, 3)

    def _genereate_contours(self, _pred):
        """
        Static method for extracting the contours from the image.
        Not to be used outside
        """
        # read image
        # img = cv2.imread(mask)

        # print(type(img),'================')
        # convert to grayscale
        # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # threshold
        # thresh = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.threshold(mask, 1.4e-5, 1, cv2.THRESH_BINARY)[1]
        ret, thresh = cv2.threshold(_pred, 1.4e-5, 1, cv2.THRESH_BINARY)
        thresh = cv2.convertScaleAbs(thresh)
        # ? Can be used to rotate all teh x_rays accordingly
        # Compute rotated bounding box
        # coords = np.column_stack(np.where(thresh > 0))
        # angle = cv2.minAreaRect(coords)[-1]

        # angle = 90 - angle
        # print(angle)
        # # Rotate image to deskew
        # (h, w) = img.shape[:2]
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # thresh = cv2.threshold(thresh,128,255,cv2.THRESH_BINARY)[1]

        # get contours
        # result = img.copy()
        contours = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Raising error in case of error in contour extraction
        if len(contours) != 2:
            raise ExtractionError("Extracted Unexpected number of contours")

        contours = contours[0] if len(contours) == 2 else contours[1]
        sorted_contours = sorted(contours, key=lambda x: len(x), reverse=True)[:2]
        # smoothened = []

        # # Loop for smoothening out the lung edges
        # # TODO To Learn more, kindly read about splines as that is what is doing the trick
        # for cntr in sorted_contours[:2]:
        #     x, y = cntr.T
        #     # Convert from numpy arrays to normal arrays
        #     x = x.tolist()[0]
        #     y = y.tolist()[0]
        #     # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        #     tck, u = splprep([x, y], u=None, s=1, per=1)
        #     # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        #     u_new = numpy.linspace(u.min(), u.max(), self.SMOOTHENING)
        #     # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        #     x_new, y_new = splev(u_new, tck, der=0)
        #     # Convert it back to numpy format for opencv to be able to display it
        #     res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        #     smoothened.append(numpy.asarray(res_array, dtype=numpy.int32))

        self.threshold_img = thresh
        self._extract_contours(sorted_contours)
        # self._extract_contours(sorted_contours)
        # return sorted_contours

    def _extract_contours(self, contours):
        cont_1 = contours[0]
        cont_2 = contours[1]

        cont_1_x = [c[0][0] for c in cont_1]
        cont_1_y = [c[0][1] for c in cont_1]

        cont_2_x = [c[0][0] for c in cont_2]
        cont_2_y = [c[0][1] for c in cont_2]

        # This is to see, out of the contours, which
        # one is for left and which one is for right
        if max(cont_1_x) > max(cont_2_x):
            self.rl["contours"] = list([cont_2_x, cont_2_y])
            self.ll["contours"] = list([cont_1_x, cont_1_y])
        else:
            self.rl["contours"] = list([cont_1_x, cont_1_y])
            self.ll["contours"] = list([cont_2_x, cont_2_y])

    def _extract_lung_dimensions(self):
        """"""

        self.lung_dimensions(self.ll["contours"], self.ll)
        self.lung_dimensions(self.rl["contours"], self.rl)

        self.smaller_lung = self.ll if self.ll["span"] < self.rl["span"] else self.rl
        self.bigger_lung = self.ll if self.smaller_lung == self.rl else self.rl

        self.threshold_img = self.process_threshold_img()
        self._lung_area()

        self._extract_cp_angle()

    def cp_angle(self):
        """
        Returns
        ---
                the angle value, and the direction of the angle with respect to parallel of the right and left cp angles respectively
        """
        angles = self._cp_angle
        return angles["right"], angles["left"]

    def cp_angle_relative_position(self):
        """
        Gives the relative position of both the cp_angles, with respect to the base x axis
        """
        right_angle = self.rl['cp_angle_position']
        left_angle = self.ll['cp_angle_position']

        return round(right_angle[1] / left_angle[1], 3 )

    def lung_dimensions(self, lung_contours, lung):
        lung_contours = list(zip(lung_contours[0], lung_contours[1]))
        lung_bottom = max(lung_contours, key=lambda x: x[1])
        lung_top = min(lung_contours, key=lambda x: x[1])
        lung_left = max(lung_contours, key=lambda x: x[0])
        lung_right = min(lung_contours, key=lambda x: x[0])
        span = lung_bottom[1] - lung_top[1]
        if lung == self.ll:
            cp_angle_position = sorted(lung_contours, key = lambda x : x[0] + x[1], reverse=True)[0]
        if lung == self.rl:
            cp_angle_position = sorted(lung_contours, key = lambda x : -x[0] + x[1], reverse=True)[0]

        data = {"top": lung_top, "bottom": lung_bottom, "left": lung_left, "right": lung_right, "span": span, 'cp_angle_position' : cp_angle_position}

        lung.update(data)

        return data

    def max_thoracic_width(self, plot=False):
        thoracic_width_array = []
        threshold_image_norm = self.threshold_img

        # Goes through the vertical span of the smaller lung and draw lines
        # and caclulates the mean of largest 10 for better answers.
        for i in range(self.smaller_lung["top"][1], self.smaller_lung["bottom"][1] + 1):
            width_array = np.where(threshold_image_norm[i, :] == 1)[0]
            if len(width_array) < 2:
                thoracic_width_array.append((i, -1))
                continue
            width = max(width_array) - min(width_array)
            thoracic_width_array.append((i, width))

        max_thoracic_width = sorted(thoracic_width_array, key=lambda x: x[1], reverse=True)
        max_thoracic_width_location_idx = max_thoracic_width[0][0]
        mean_max_thoracic_width = np.mean([y for _, y in max_thoracic_width[:10]])

        if plot:
            plt.plot((0, 255), (max_thoracic_width_location_idx, max_thoracic_width_location_idx))

        return max_thoracic_width_location_idx, mean_max_thoracic_width

    def plot_contours(self, plot_middle=False, plot_grid=False, plot_cp=False, plot_max_thorax=False):

        plt.plot(self.ll["contours"][0], self.ll["contours"][1], "g")
        plt.plot(self.rl["contours"][0], self.rl["contours"][1], "r")
        plt.ylim(256, 0)
        plt.xlim(0, 256)

        if plot_middle:
            mid_point = self.mid_point()
            plt.plot((mid_point, mid_point), (0, 256))

        if plot_max_thorax:
            self.max_thoracic_width(plot=True)

        axes = plt.gca()
        axes.set_aspect(1)
        plt.grid(visible=plot_grid, which="both")

    def mid_point(self):
        mid_point = self.ll["right"][0] - (self.ll["right"][0] - self.rl["left"][0]) // 2
        return mid_point

    def process_threshold_img(self):
        """
        Removing extra random pixels from the threshold image
        before processing it further
        """
        t_img = self.threshold_img
        r_point = self.rl["right"][0]
        l_point = self.ll["left"][0]
        top = (
            self.bigger_lung["top"][1]
            if self.bigger_lung["top"][1] < self.smaller_lung["top"][1]
            else self.smaller_lung["top"][1]
        )
        bottom = (
            self.bigger_lung["bottom"][1]
            if self.bigger_lung["bottom"][1] > self.smaller_lung["bottom"][1]
            else self.smaller_lung["bottom"][1]
        )

        t_img[:, :r_point] = 0
        t_img[:, l_point:] = 0
        t_img[:top, :] = 0
        t_img[bottom:, :] = 0
        return t_img

    def _lung_area(self):
        mid = self.mid_point()
        thresh = self.threshold_img
        self.ll["area"] = np.sum(thresh[:, mid:])
        self.rl["area"] = np.sum(thresh[:, :mid])

    def _extract_cp_angle(self):
        r, l = self.contours_polar

        # Right
        cp_angle_point_right = sorted(r, key=lambda x: -x[0] + 2 * x[1], reverse=True)[0]
        cp_angle_point_right_index = r.index(cp_angle_point_right)
        r_cp = self.__helper_cp_angle(r, cp_angle_point_right, cp_angle_point_right_index)

        # Left
        # Getting the cp angles, by sorting them and giving the y direction more weight, so
        # as to get the angle with more certainity
        cp_angle_point_left = sorted(l, key=lambda x: x[0] + 2 * x[1], reverse=True)[0]
        cp_angle_point_left_index = l.index(cp_angle_point_left)
        l_cp = self.__helper_cp_angle(l, cp_angle_point_left, cp_angle_point_left_index)

        self._left_cp_angle_coordinates = cp_angle_point_left

        self._cp_angle = {"left": (l_cp[0], l_cp[1]), "right": (r_cp[0], r_cp[1])}

    def __helper_cp_angle(self, contour, cp_angle_point, max_idx, plot=True):
        """
        For extracting the cp angle for a given point, after we have extracted the cp points
        of either the left or right lung
        """

        # Taking a mean of the before and after points so as to be more reliable
        # * If you feel the answers are not coming out correctly, you can maybe loop over
        # * a few angles and take their mean.
        range_idx = self.CP_ANGLE_RANGE
        mean_before_point = np.mean([contour[max_idx - i] for i in range_idx], 0)
        mean_after_point = np.mean([contour[max_idx + i] for i in range_idx], 0)

        # Bringing vector's origins to the cp_angle
        vector_1 = np.subtract(mean_before_point, cp_angle_point)
        vector_2 = np.subtract(mean_after_point, cp_angle_point)

        # Unit vectors
        unit_vector1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector2 = vector_2 / np.linalg.norm(vector_2)

        angle = calc_angle(unit_vector1, unit_vector2)

        # As there was discrepancy in the axis of images, y starting from top,
        # Had to manually caclulate it, and get the bisecting angle
        # TODO In future, maybe can find a better way of doing this
        x2, y2 = unit_vector2
        x1, y1 = unit_vector1
        bisecting_vector = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        cp_angle_unit_vector = bisecting_vector / np.linalg.norm(bisecting_vector)

        # Making sure just in case that the angle is below parallel
        cp_vector_angle = abs(calc_angle(cp_angle_unit_vector, (1, 0)))

        if cp_vector_angle > 90:
            cp_vector_angle = 180 - cp_vector_angle

        # if plot:
        #     plot_point(mean_before_point, 'ko')
        #     plot_point(mean_after_point, 'ro')
        #     plot_point(100 * cp_angle_unit_vector,
        #                'c->', origin=cp_angle_point)

        return angle, cp_vector_angle

    def report(self):
        print("=================================================")
        print("Vertical Span Ratio (Right vs Left) = ", self.lung_vertical_span_ratio)
        print("Lung Area Ratio (Right vs Left) = ", self.area_ratio)
        print("Height-to-Width Ratio of the thorax = ", self.height_width_ratio)
        print("Max Thoracic Width = ", self.max_thoracic_width()[1])
        print("CT Ratio = ", self.ct_ratio()[0])
        print("CT Area = ", self.ct_ratio()[1])
        print("------------------------")
        print("Cp Angles")
        print(f"Right -> Angle Size = {self.cp_angle()[0][0]} Angle Direction = {self.cp_angle()[0][1]}")
        print(f"Left -> Angle Size = {self.cp_angle()[1][0]} Angle Direction = {self.cp_angle()[1][1]}")
        print("=================================================")

    def plot_mask(self):
        plt.imshow(self.threshold_img, cmap="gray")

    def data_classification(self):
        numeric_data = {
            'ct_ratio' : tf.convert_to_tensor([self.ct_ratio()[0]]),
            'ct_area' : tf.convert_to_tensor([self.ct_ratio()[1]]),
            'right_cp_angle': tf.convert_to_tensor([self.cp_angle()[0][0]]),
            'right_cp_angle_direction' : tf.convert_to_tensor([self.cp_angle()[0][1]]),
            'left_cp_angle': tf.convert_to_tensor([self.cp_angle()[1][0]]),
            'left_cp_angle_direction': tf.convert_to_tensor([self.cp_angle()[1][1]]),
            'cp_rel_position': tf.convert_to_tensor([self.cp_angle_relative_position()]),
            'height_width_ratio' : tf.convert_to_tensor([self.height_width_ratio]),
            'area_ratio': tf.convert_to_tensor([self.area_ratio]),
            'vertical_span_ratio': tf.convert_to_tensor([self.lung_vertical_span_ratio])
        }
        return numeric_data


DATA = ['ct_ratio','ct_area','right_cp_angle',
                  'right_cp_angle_direction','left_cp_angle','left_cp_angle_direction',
                  'cp_rel_position','height_width_ratio','area_ratio','vertical_span_ratio']

def plot_point(point, color):
    plt.plot(point[0], point[1], color)


def calc_angle(v1, v2):
    dot_prod = np.dot(v1, v2)
    angle_rad = np.arccos(np.clip(dot_prod, -1, 1))

    # The final angle between the two vectors
    angle = np.rad2deg(angle_rad)

    return angle
