import cv2
import time

import numpy as np

from src.strategies import detection_strategy as ds
from src.strategies import angle_calculation_strategy as acs

class AngleVisualizer:
    def __init__(self, detection_model_path, keypoints, imgsz=512):
        self.detector = ds.YOLOStrategy(imgsz=imgsz, weights_path=detection_model_path).create_model()
        self.angle_calculator = acs.Angle2DCalculation()
        self.keypoints = keypoints

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.COLORS = {
            'white': (255, 255, 255),
            'pink': (229, 156, 209),
            'light_green': (100, 233, 127),
            'purple': (128, 0, 128)
        }

    def process_video(self, video_path, output_path, show_fps=False):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #  Используем  кодек  mp4v
        out = cv2.VideoWriter(output_path, fourcc, fps, (width,  height))  #  Создаем  VideoWriter

        pTime = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detector.process_frame(frame, plot=False, device='mps', verbose=False)
            keypoints = self.detector.get_coordinates()

            if len(keypoints):
                keypoint_coords = self.get_keypoint_coords(keypoints)
                self.draw_skeleton(frame, keypoint_coords)
                self.calculate_and_display_angles(frame, keypoint_coords)

            if show_fps:
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(frame, f'fps: {int(fps)}', (10, 30), self.font, 1, self.COLORS['white'], 2)

            out.write(frame)  #  Записываем  кадр  в  выходной  файл
            cv2.imshow("Angle Visualization", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()  #  Закрываем  VideoWriter
        cv2.destroyAllWindows()

    def get_keypoint_coords(self, keypoints):
        keypoint_coords = {}
        for feature_name, feature_keypoints in self.keypoints.items():
            if feature_name == "nose":
                keypoint_coords["nose"] = self.detector.get_landmark_coordinates("nose")
            else:
                coords = self.detector.get_landmark_coordinates(feature_name)
                for i, keypoint in enumerate(feature_keypoints):
                    keypoint_coords[f"{feature_name}_{keypoint}"] = coords[i]
        return keypoint_coords

    def draw_skeleton(self, frame, keypoint_coords):
        #  Отрисовка  скелета  на  основе  keypoints
        dominant_side = self.get_dominant_side(keypoint_coords)
        for keypoint_type in self.keypoints[dominant_side]:
            keypoint_name = f"{dominant_side}_{keypoint_type}"
            if keypoint_name in keypoint_coords:
                cv2.circle(frame, keypoint_coords[keypoint_name], 7, self.COLORS['white'], -1, lineType=self.linetype)
                if keypoint_type == "hip":
                    prev_keypoint_name = f"{dominant_side}_shoulder"
                    self.draw_landmark_line(frame, keypoint_coords[prev_keypoint_name],
                                            keypoint_coords[keypoint_name], self.COLORS['pink'], 4)
                elif keypoint_type != self.keypoints[dominant_side][0]:
                    prev_keypoint_name = f"{dominant_side}_{self.keypoints[dominant_side][self.keypoints[dominant_side].index(keypoint_type) - 1]}"
                    self.draw_landmark_line(frame, keypoint_coords[prev_keypoint_name],
                                            keypoint_coords[keypoint_name], self.COLORS['pink'], 4)

    def calculate_and_display_angles(self, frame, keypoint_coords):
        # Расчет и отображение углов между ключевыми точками
        dominant_side = self.get_dominant_side(keypoint_coords)

        # Список углов, которые нужно рассчитать
        angles_to_calculate = [
            ("wrist", "elbow", "shoulder"),  # wrist_elbow_shoulder
            ("elbow", "shoulder", "hip"),  # elbow_shoulder_hip
            ("elbow", "shoulder", None),  # shoulder_vertical
            ("shoulder", "hip", None),  # hip_vertical
            ("shoulder", "hip", "knee"),  # shoulder_hip_knee
            ("hip", "knee", None),  # knee_vertical
            ("hip", "knee", "ankle"),  # hip_knee_ankle
            ("knee", "ankle", None)  # ankle_vertical
        ]

        #  Расчет и отрисовка углов
        for point1, point2, point3 in angles_to_calculate:
            p1 = keypoint_coords[f"{dominant_side}_{point1}"]
            if point2 is not None:
                p2 = keypoint_coords[f"{dominant_side}_{point2}"]
            if point3 is not None:
                p3 = keypoint_coords[f"{dominant_side}_{point3}"]

            if point3 is None:  # Вертикальный угол
                angle = self.angle_calculator.calculate_angle(p2, np.array([p2[0], 0]), p1)
                cv2.putText(frame, str(int(angle)), (p1[0] - 40, p1[1]),
                            self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.line(frame, (p1[0], p1[1] - 50), (p1[0], p1[1] + 50), self.COLORS['purple'], 2)
            else:  # Обычный  угол
                angle = self.angle_calculator.calculate_angle(p1, p3, p2)
                cv2.putText(frame, str(int(angle)), (p2[0] + 10, p2[1]),
                            self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

    def get_dominant_side(self, keypoint_coords):
        point = 'elbow'
        return "left" if keypoint_coords[f"left_{point}"][1] > keypoint_coords[f"right_{point}"][1] else "right"

    def draw_landmark_line(self, frame, p1, p2, color, thickness):
        if not np.array_equal(p1, [0, 0]) and not np.array_equal(p2, [0, 0]):
            cv2.line(frame, p1, p2, color, thickness, lineType=self.linetype)
        return frame


if __name__ == "__main__":
    detection_model_path = "../models/weights/yolov8s-pose.pt"
    keypoints = {
        "left": ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"],
        "right": ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"],
        "nose": ["nose"]
    }
    visualizer = AngleVisualizer(detection_model_path, keypoints)
    video_path = r'/Users/egorken/Downloads/bicep curls good.mp4'
    # video_path = r'/Users/egorken/Downloads/x2mate.com-How to do a Dumbbell Hammer Curl.mp4'
    print('Processing video...')
    visualizer.process_video(video_path, "output.mp4", show_fps=False)
