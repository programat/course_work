# pose_processor.py

from abc import ABC, abstractmethod
import time
import cv2
import numpy as np
from src.strategies import angle_calculation_strategy as acs
from src.strategies import detection_strategy as dc
from src.models import exercise as exr

from src.models import opencv_elements


class PoseProcessor:
    def __init__(self, detection_strategy: dc.DetectionStrategy,
                 angle_calculation_strategy: acs.AngleCalculationStrategy, level=0):

        self.detector = detection_strategy
        self.angle_calculation = angle_calculation_strategy

        self.cv_elem = opencv_elements.OpenCVElements

        # Font type
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # line type
        self.linetype = cv2.LINE_AA
        # set radius to draw arc
        self.radius = 30
        self.COLORS = {
            'black': (14, 16, 15),
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255),
            'purple': (143, 126, 213),
            'pink': (229, 156, 209)
        }


    @abstractmethod
    def process(self, frame: np.array, curls):
        pass