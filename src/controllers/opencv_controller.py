# opencv_controller.py

import time
import traceback

import cv2
from src.strategies.pose_processor import squats_processor, dumbbell_processor

from src.strategies import exercise_processor


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

    def setup(self, stream=0, level=0, video_path=None):
        if stream:
            self.vid = cv2.VideoCapture(1)
        else:
            self.vid = cv2.VideoCapture(video_path)
            # self.vid = cv2.VideoCapture('/Users/egorken/Downloads/How to bodyweight squat.mp4')
            # self.vid = cv2.VideoCapture('/Users/egorken/Downloads/10 Min Squat Workout with 10 Variations - No Repeats No Talking.mp4')
            # self.vid = cv2.VideoCapture('/Users/egorken/Downloads/bicep curls.mp4')

        if self.selected_exercise is None or \
                self.angle_calculation_strategy is None or \
                self.detection_strategy is None:
            raise ValueError("All required components isn't installed. Install them first.")

        # setting chosen strategies
        self.set_detection_strategy(self.detection_strategy)
        self.set_angle_calculation_strategy(self.angle_calculation_strategy)

        # choosing pose_processor strategy
        if self.selected_exercise == "Squats":
            self.set_pose_processor_strategy(squats_processor.SquatsProcessor(self.detection_strategy, self.angle_calculation_strategy, level))
        elif self.selected_exercise == "Dumbbell":
            self.set_pose_processor_strategy(dumbbell_processor.DumbbellProcessor(self.detection_strategy, self.angle_calculation_strategy, level))
        else:
            print(f"Unknown exercise: {self.selected_exercise}")
            # raise ValueError(f"Unknown exercise: {self.selected_exercise}")
        return self

    def process(self, show_fps=False, curls=None, plot=False):
        pTime = 0

        self.set_pose_processor_strategy(exercise_processor.ExerciseProcessor(self.detection_strategy, self.angle_calculation_strategy, '../strategies/dumbbell.json'))

        try:
            while self.vid.isOpened():
                _, self.frame = self.vid.read()
                # TODO: device mps:0 while
                # (TypeError: can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.)
                self.frame = self.detection_strategy.process_frame(self.frame, plot=plot, device='mps')

                try:
                    self.pose_processor.process(self.frame, curls=curls)
                except Exception as ex:
                    print(f'pose_processor exception: {ex}')
                    traceback.print_exc()
                    # exit()


                if show_fps:
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    cv2.putText(self.frame, f'fps: {int(fps)}', (1180, 45), cv2.FONT_HERSHEY_PLAIN, 1.2,
                                 (255, 255, 255), 2)

                cv2.imshow(f'AI Trainer: {self.selected_exercise} training', self.frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except cv2.error as ex:
            print(f'opencv_controller: {ex}')

        self.vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    from src.strategies import detection_strategy, angle_calculation_strategy

    detector = detection_strategy.YOLOStrategy(imgsz=512, weights_path='/Users/egorken/PycharmProjects/course work/src/models/weights/yolov8m-pose.pt').create_model()
    angle = angle_calculation_strategy.Angle2DCalculation()
    chosen_exercise = 'Test'

    opencv_controller = OpenCVController(detector, angle, chosen_exercise)
    # video_path = r'/Users/egorken/Downloads/10 Min Squat Workout with 10 Variations - No Repeats No Talking.mp4'
    # video_path = r'/Users/egorken/Downloads/How to bodyweight squat.mp4'
    video_path = r'/Users/egorken/Downloads/bicep curls.mp4'
    opencv_controller.setup(stream=0, video_path=video_path)
    opencv_controller.process(curls=20, plot=False, show_fps=True)