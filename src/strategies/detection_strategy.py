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
    @abstractmethod
    def get_landmark_coordinates(self, feature):
        pass
    @abstractmethod
    def is_plotted(self):
        pass
    @abstractmethod
    def change_parameters(self, path='', imgsz=-1, conf=-1, iou=-1):
        pass


class YOLOStrategy(DetectionStrategy):
    def __init__(self, imgsz=320, weights_path=os.path.join(os.path.dirname(__file__), r'../models/weights/yolov8s-pose.pt'), conf=0.25, iou=0.7,):
        self.weights_path = weights_path
        self.imgsz = imgsz,
        self.conf = conf,
        self.iou = iou
        self.model = None
        self._is_plotted = False

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

    def change_parameters(self, path='', imgsz=-1, conf=-1, iou=-1):
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
        self.model = YOLO(self.weights_path)
        return self

    def process_frame(self, frame, verbose=False, device='cpu', plot=False):
        self.is_plotted=plot
        self.results = self.model(frame, verbose=verbose, device=device, imgsz=self.imgsz)
        if plot:
            annotated_frame = self.results[0].plot(labels=False, boxes=False)
            return annotated_frame
        else:
            return frame
        # тут обдумать трек для нескольких людей

    def get_coordinates(self):
        if self.results[0].keypoints.xy.device.type == 'mps':
            res_coord = [r.keypoints.xy.to(int).cpu().numpy() for r in self.results]
        else:
            res_coord = [r.keypoints.xy.to(int).numpy() for r in self.results]
        return res_coord[0]

    def get_landmark_features(self):
        return self.landmark_features_dict

    def get_landmark_coordinates(self, feature):
        if feature == 'nose':
            return self.get_coordinates()[0][self.landmark_features_dict[feature]]
        if feature.lower() in ['left', 'right']:
            # return self.get_coordinates()[0][self.landmark_features_dict[feature].values()]

            shldr_coord = self.get_coordinates()[0][self.landmark_features_dict[feature]['shoulder']]
            elbow_coord = self.get_coordinates()[0][self.landmark_features_dict[feature]['elbow']]
            wrist_coord = self.get_coordinates()[0][self.landmark_features_dict[feature]['wrist']]
            hip_coord = self.get_coordinates()[0][self.landmark_features_dict[feature]['hip']]
            knee_coord = self.get_coordinates()[0][self.landmark_features_dict[feature]['knee']]
            ankle_coord = self.get_coordinates()[0][self.landmark_features_dict[feature]['ankle']]

            return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord
        else:
            raise ValueError('Feature needs to be "nose", "left" or "right"')

    def is_plotted(self):
        return self._is_plotted



class YoloNasStrategy(DetectionStrategy):
    def process_frame(self, frame):
        # Реализация обработки изображения с использованием YOLO-NAS
        pass


class MPStrategy(DetectionStrategy):
    def process_frame(self, frame):
        # Реализация обработки изображения с использованием MediaPipe
        pass
