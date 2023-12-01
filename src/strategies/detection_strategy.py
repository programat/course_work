# detection_strategy.py

from abc import ABC, abstractmethod
from ultralytics import YOLO
import cv2
import numpy as np

class DetectionStrategy(ABC):
    @abstractmethod
    def process_image(self, frame):
        pass
    # def get_coordinates(self):
    #     pass


class YOLOStrategy(DetectionStrategy):
    def __init__(self):
        self.model = YOLO('../src/models/weights/yolov8m-pose.pt')

    def get_coordinates(self):
        pass

    def process_image(self, frame):
        # Реализация обработки изображения с использованием YOLOv8
        pass

class YoloNasStrategy(DetectionStrategy):
    def process_image(self, frame):
        # Реализация обработки изображения с использованием YOLO-NAS
        pass


class MPStrategy(DetectionStrategy):
    def process_image(self, frame):
        # Реализация обработки изображения с использованием MediaPipe
        pass
