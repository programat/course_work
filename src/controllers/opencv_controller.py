# opencv_controller.py

import time
import cv2
from src.strategies import pose_processor_strategy as pps


# В opencv_controller.py
class OpenCVController:
    def __init__(self, detection_strategy=None,  angle_calculation_strategy=None,
                 selected_exercise=None):
        self.angle_calculation_strategy = angle_calculation_strategy
        self.detection_strategy = detection_strategy
        self.selected_exercise = selected_exercise
        self.pose_processor = None

    def set_selected_exercise(self, exercise_name):
        self.selected_exercise = exercise_name

    def set_angle_calculation_strategy(self, strategy):
        self.angle_calculation_strategy = strategy

    def set_detection_strategy(self, strategy):
        self.detection_strategy = strategy

    def set_pose_processor_strategy(self, strategy):
        self.pose_processor = strategy

    def setup(self, stream=0, level=0, video_source=0):
        if stream:
            self.vid = cv2.VideoCapture(1)
        else:
            self.vid = cv2.VideoCapture('/Users/egorken/Downloads/How to bodyweight squat.mp4')

        if self.selected_exercise is None or \
                self.angle_calculation_strategy is None or \
                self.detection_strategy is None:
            raise ValueError("All required components isn't installed. Install them first.")

        # setting chosen strategies
        self.set_detection_strategy(self.detection_strategy)
        self.set_angle_calculation_strategy(self.angle_calculation_strategy)

        # choosing pose_processor strategy
        if self.selected_exercise == "Squats":
            self.set_pose_processor_strategy(pps.SquatsProcessor(self.detection_strategy, self.angle_calculation_strategy))
        elif self.selected_exercise == "Dumbbell":
            pass
        else:
            raise ValueError(f"Unknown exercise: {self.selected_exercise}")
        return self

    def process(self, show_fps=False, curls=None):
        pTime = 0

        try:
            while self.vid.isOpened():
                _, self.frame = self.vid.read()
                self.frame = self.detection_strategy.process_frame(self.frame, plot=False)

                try:
                    self.pose_processor.process(self.frame, curls=curls)
                except Exception as ex:
                    print(ex)


                if show_fps and self.vid.isOpened():
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    cv2.putText(self.frame, f'fps: {int(fps)}', (1160, 60), cv2.FONT_HERSHEY_PLAIN, 1.2,
                                 (255, 255, 255), 2)

                    cv2.imshow('AI Trainer: Squats training', self.frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except cv2.error:
            pass

        self.vid.release()
        cv2.destroyAllWindows()