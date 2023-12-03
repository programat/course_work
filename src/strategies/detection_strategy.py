# detection_strategy.py

from abc import ABC, abstractmethod
from ultralytics import YOLO
import os

class DetectionStrategy(ABC):
    @abstractmethod
    def process_frame(self, frame):
        pass
    @abstractmethod
    def get_coordinates(self):
        pass
    @abstractmethod
    def get_landmark_features(self):
        pass
    @abstractmethod
    def create_model(self):
        pass
    @abstractmethod
    def process_frame(self, frame):
        pass


class YOLOStrategy(DetectionStrategy):
    def __init__(self, imgsz=320, weights_path=os.path.join(os.path.dirname(__file__), r'.../models/weights/yolo8s-pose.pt'), conf=0.25, iou = 0.7,):
        self.weights_path = weights_path
        self.imgsz = imgsz,
        self.conf = conf,
        self.iou = iou
        self.model = None

        # Dictionary to maintain the various landmark features.
        self.landmark_features_dict = {}
        self.landmark_features_dict_left = {
            'shoulder': 5,
            'elbow': 7,
            'wrist': 9,
            'hip': 11,
            'knee': 13,
            'ankle': 15,
            'foot': None
        }

        self.landmark_features_dict_right = {
            'shoulder': 6,
            'elbow': 8,
            'wrist': 10,
            'hip': 12,
            'knee': 14,
            'ankle': 16,
            'foot': None
        }

        self.landmark_features_dict['left'] = self.landmark_features_dict_left
        self.landmark_features_dict['right'] = self.landmark_features_dict_right
        self.landmark_features_dict['nose'] = 0

    def change_parameters(self, imgsz, path, conf, iou):
        if path != '':
            self.path_weights = path

        if imgsz != -1:
            self.imgsz = imgsz

        if conf != -1:
            self.conf = conf

        if iou != -1:
            self.iou = iou

        print(self.path_weights, self.imgsz, self.conf, self.iou)

    def create_model(self):
        if self.model is not None:
            del self.model
        self.model = YOLO(self.path_weights)
        return self

    def process_frame(self, frame, verbouse=False, device='cpu'):
        self.results = self.model(frame, verbouse=verbouse, device=device, imgsz=self.imgsz)
        annotated_frame = self.results[0].plot(labels=False, boxes=False)
        return annotated_frame

    def get_coordinates(self):
        res_coord = [r.keypoints.xy.to(int).numpy() for r in self.results]
        return res_coord

    def get_landmark_features(self):
        return self.landmark_features_dict


class YoloNasStrategy(DetectionStrategy):
    def process_frame(self, frame):
        # Реализация обработки изображения с использованием YOLO-NAS
        pass


class MPStrategy(DetectionStrategy):
    def process_frame(self, frame):
        # Реализация обработки изображения с использованием MediaPipe
        pass
