# opencv_controller.py

import time
import cv2
from src.strategies import angle_calculation_strategy as acs
from src.strategies import detection_strategy as dc
from src.strategies import pose_processor_strategy as pps


# В opencv_controller.py
class OpenCVController:
    def __init__(self):
        self.angle_calculation_strategy = None
        self.detection_strategy = None
        self.pose_processor_strategy = None
        self.selected_exercise = None

    def set_selected_exercise(self, exercise_name):
        self.selected_exercise = exercise_name

    def set_angle_calculation_strategy(self, strategy):
        self.angle_calculation_strategy = strategy

    def set_detection_strategy(self, strategy):
        self.detection_strategy = strategy

    def set_pose_processor_strategy(self, strategy):
        self.pose_processor_strategy = strategy

    def process_frame(self, frame):
        if self.selected_exercise is None or \
           self.angle_calculation_strategy is None or \
           self.detection_strategy is None or \
           self.pose_processor_strategy is None:
            raise ValueError("Не установлены все необходимые компоненты. Установите их перед обработкой.")


        # Выбор стратегии вычисления углов
        angle_calculation_strategy = acs.Angle2DCalculation()

        # Выбор стратегии распознавания
        detection_strategy = dc.YOLOStrategy()

        # Установка выбранных стратегий
        self.set_detection_strategy(detection_strategy)
        self.set_angle_calculation_strategy(angle_calculation_strategy)

        # Выбор стратегии распознавания
        if self.selected_exercise == "squats":
            squats_processor = pps.SquatsProcessor(detection_strategy, angle_calculation_strategy)
            self.set_pose_processor_strategy(squats_processor)
        elif self.selected_exercise == "dumbbell":
            pass
        else:
            raise ValueError(f"Неизвестное упражнение: {self.selected_exercise}")

        # Выполнение действий с использованием выбранных стратегий
        angles = self.angle_calculation_strategy.calculate_angles(frame)
        detections = self.detection_strategy.detect_objects(frame)
        processed_pose = self.pose_processor_strategy.process_pose(detections, angles)

        # Возможно, дополнительная логика обработки результатов

        return processed_pose
