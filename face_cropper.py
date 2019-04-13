import dlib
import cv2
from imutils import face_utils, rotate
from math import sqrt, pi, atan2
import numpy as np


class FaceCropper:
    def __init__(self, shape_predictor, height=None, width=None, ratio=None,
                 color_treshold=255, angle_treshold=5):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = shape_predictor
        if ratio:
            if not height and width:
                self.height = int(width * ratio)
                self.width = width
                self.ratio = ratio
            elif not width and height:
                self.width = int(height / ratio)
                self.height = height
                self.ratio = ratio
            else:
                self.height = height
                self.width = width
                self.ratio = height / width
        else:
            self.height = height
            self.width = width
            if width:
                self.ratio = height / width
            else:
                self.ratio = None
        self.color_treshold = color_treshold
        self.angle_treshold = angle_treshold
        self.cached = None
        self.cached_greyscale = None

    def update_cache(self, image):
        del self.cached
        del self.cached_greyscale
        self.cached = image
        self.cached_greyscale = None

    def get_grayscale(self):
        if self.cached_greyscale:
            return self.cached_greyscale
        else:
            return cv2.cvtColor(self.cached, cv2.COLOR_BGR2GRAY)

    def crop(self, image):
        self.update_cache(image)
        try:
            self.expand()
            contours = [self.approximate_contour(contour) for contour in
                        self.find_contours()]
            self.find_face(max(contours, key = cv2.contourArea))
            self.align_face()
            box = self.find_face_box()
            self.update_cache(cv2.copyMakeBorder(self.cached, 20, 0,
                                                 0, 0,
                                                 cv2.BORDER_CONSTANT,
                                                 value=[255, 255, 255]))
            self.resize(box)
        except Exception:
            pass
        return self.cached

    def expand(self):
        height, width = self.cached.shape[:2]
        vertical = int((sqrt(2) - 1) * height)
        horizontal = int((sqrt(2) - 1) * width)
        self.update_cache(cv2.copyMakeBorder(self.cached, vertical, vertical,
                                             horizontal, horizontal,
                                             cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255]))

    def find_contours(self):
        blured = cv2.blur(self.get_grayscale(), (15,15))
        ret, tresh = cv2.threshold(blured, self.color_treshold, 255,
                                   cv2.THRESH_BINARY_INV)
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, kernel, iterations=5)
        img, contours, hierarchy = cv2.findContours(opening,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def approximate_contour(self, contour):
        epsilon = 0.01*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def find_face(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        x -= 0.7 * w
        y -= 0.4 * h
        w *= 2.4
        h *= 1.8
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        self.update_cache(self.cached[y:y + h, x:x + w])

    def align_face(self):
        rect = self.detector(self.get_grayscale(), 1)[0]
        shape = face_utils.shape_to_np(self.predictor(self.get_grayscale(),
                                                     rect))
        # 37 - 42 are points for left eye
        left = np.median(shape[37:43], axis=0)
        # 43 - 48 are points for right eye
        right = np.median(shape[43:49], axis=0)
        angle = atan2(left[1] - right[1], right[0] - left[0]) * 180 / pi
        if abs(angle) > self.angle_treshold:
            self.update_cache(rotate(self.cached, -angle))

    def find_face_box(self):
        box = self.detector(self.get_grayscale(), 1)[0]
        rect = self.detector(self.get_grayscale(), 1)[0]
        shape = face_utils.shape_to_np(self.predictor(self.get_grayscale(),
                                                     rect))
        left_eye = np.median(shape[37:43], axis=0)
        right_eye = np.median(shape[43:49], axis=0)
        central_x = np.mean([left_eye, right_eye, shape[9], shape[28],
                            shape[29], shape[30], shape[31], shape[34],
                            shape[58], shape[63], shape[67]], axis=0)[0]
        right, left = box.right(), box.left()
        top, bottom = box.top(), box.bottom()
        height, width = box.height(), box.width()
        # these are magical coefficients aquired by practical search
        # change only if you know what you are doing
        height *= 1.1
        width *= 1.1
        top -= 0.55*height
        bottom += 0.0*height
        left -= 0.2*width
        right += 0.2*width
        fix = central_x - (right + left) * 0.5
        left += fix
        right += fix
        contours = [self.approximate_contour(contour) for contour in
                    self.find_contours()]
        contour = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(contour)
        top = int(top)
        top -= 0.1*height
        return (int(left), int(top), int(right), int(bottom))

    def resize(self, box):
        left, top, right, bottom = box
        height = bottom - top
        width = right - left
        if self.ratio:
            if height/width < self.ratio:
                expand = self.ratio * width - height
                bottom += expand
                bottom = round(bottom)
            elif height/width > self.ratio:
                expand = height / self.ratio - width
                left -= expand * 0.5
                right += expand * 0.5
                left = round(left)
                right = round(right)
        self.update_cache(self.cached[top:bottom, left:right])
        if self.width and self.height:
            self.update_cache(cv2.resize(self.cached, (self.width, self.height),
                              interpolation = cv2.INTER_AREA))
        elif self.width:
            ratio = self.width / width
            self.update_cache(cv2.resize(self.cached,
                                         (self.width, round(height * ratio)),
                                         interpolation = cv2.INTER_AREA))
        elif self.height:
            ratio = self.height / height
            self.update_cache(cv2.resize(self.cached,
                                         (round(width * ratio), self.height),
                                         interpolation = cv2.INTER_AREA))
        elif self.ratio:
            self.update_cache(cv2.resize(self.cached,
                                         (round(width * self.ratio), height),
                                         interpolation = cv2.INTER_AREA))
