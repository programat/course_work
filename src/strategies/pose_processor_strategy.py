# pose_processor_strategy.py

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

    @abstractmethod
    def process_pose(self):
        pass

class SquatsProcessor(PoseProcessor):
    def __init__(self, detection_strategy: dc.DetectionStrategy, angle_calculation_strategy: acs.AngleCalculationStrategy, level=0):
        super().__init__(detection_strategy, angle_calculation_strategy, level)

        self.exercise = exr.SquatExercise(level)
        self.thresholds = self.exercise.get_thresholds()

        # --- сомнительно, ну окэй

        # Font type
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # line type
        self.linetype = cv2.LINE_AA
        # set radius to draw arc
        self.radius = 30
        self.COLORS = {
                        'blue'       : (0, 127, 255),
                        'red'        : (255, 50, 50),
                        'green'      : (0, 255, 127),
                        'light_green': (100, 233, 127),
                        'yellow'     : (255, 255, 0),
                        'magenta'    : (255, 0, 255),
                        'white'      : (255, 255, 255),
                        'cyan'       : (0, 255, 255),
                        'light_blue' : (102, 204, 255)
                      }
        self.landmark_features_dict = detection_strategy.get_landmark_features()
        self.state_tracker = {
            'state_seq': [],

            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,

            # 0 --> Bend Backwards, 1 --> Bend Forward, 2 --> Keep shin straight, 3 --> Deep squat
            'DISPLAY_TEXT': np.full((4,), False),
            'COUNT_FRAMES': np.zeros((4,), dtype=np.int64),

            'LOWER_HIPS': False,

            'INCORRECT_POSTURE': False,

            'prev_state': None,
            'curr_state':None,

            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT': 0
        }

        self.FEEDBACK_ID_MAP = {
            0: ('BEND BACKWARDS', 215, (0, 153, 255)),
            1: ('BEND FORWARD', 215, (0, 153, 255)),
            2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
            3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
        }

    # def set_angle_calculation_strategy(self, angle_calculation_strategy: acs.AngleCalculationStrategy):
    #     self.angle_calculation_strategy = angle_calculation_strategy

    def calculate_angle(self, landmarks):
        return self.angle_calculation.calculate_angles(landmarks)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):

        if lower_hips_disp:
            self.cv_elem.draw_text(
                frame,
                'LOWER YOUR HIPS',
                pos=(30, 80),
                text_color=(0, 0, 0),
                font_scale=0.6,
                text_color_bg=(255, 255, 0)
            )

        for idx in np.where(c_frame)[0]:
            self.cv_elem.draw_text(
                frame,
                dict_maps[idx][0],
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=dict_maps[idx][2]
            )

        return frame

    def process_pose(self):
        pass

    def process(self, frame: np.array):
        play_sound = None

        frame_height, frame_width, _ = frame.shape

        self.detector.process_frame(frame)
        # Process the image.
        keypoints = self.detector.get_coordinates()

        if len(keypoints):
            nose_coord = self.detector.get_landmark_coordinates('nose')
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord = \
                self.detector.get_landmark_coordinates('left')
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord = \
                self.detector.get_landmark_coordinates('right')

            offset_angle = self.angle_calculation.calculate_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']:

                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                # if self.flip_frame:
                #     frame = cv2.flip(frame, 1)

                if display_inactivity:
                    # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 90),
                    #             self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                self.cv_elem.draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']),
                    pos=(int(frame_width * 0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                self.cv_elem.draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']),
                    pos=(int(frame_width * 0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0)
                )

                self.cv_elem.draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, frame_height - 60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                self.cv_elem.draw_text(
                    frame,
                    'OFFSET ANGLE: ' + str(offset_angle),
                    pos=(30, frame_height - 30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                dist_l_sh_hip = abs(left_ankle_coord[1] - left_shldr_coord[1])
                dist_r_sh_hip = abs(right_ankle_coord[1] - right_shldr_coord)[1]

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord

                    multiplier = -1

                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord

                    multiplier = 1

                # ------------------- Verical Angle calculation --------------

                hip_vertical_angle = self.angle_calculation.calculate_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier * hip_vertical_angle,
                            color=self.COLORS['white'], thickness=3, lineType=self.linetype)

                self.cv_elem.draw_dotted_line(frame, hip_coord, start=hip_coord[1] - 80, end=hip_coord[1] + 20,
                                 line_color=self.COLORS['blue'])

                knee_vertical_angle = self.angle_calculation.calculate_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20),
                            angle=0, startAngle=-90, endAngle=-90 - multiplier * knee_vertical_angle,
                            color=self.COLORS['white'], thickness=3, lineType=self.linetype)

                self.cv_elem.draw_dotted_line(frame, knee_coord, start=knee_coord[1] - 50, end=knee_coord[1] + 20,
                                 line_color=self.COLORS['blue'])

                ankle_vertical_angle = self.angle_calculation.calculate_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier * ankle_vertical_angle,
                            color=self.COLORS['white'], thickness=3, lineType=self.linetype)

                self.cv_elem.draw_dotted_line(frame, ankle_coord, start=ankle_coord[1] - 50, end=ankle_coord[1] + 20,
                                 line_color=self.COLORS['blue'])

                # ------------------------------------------------------------

                # # # Join landmarks.
                # cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                # cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                # cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                # cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                # cv2.line(frame, ankle_coord, knee_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                #
                # # Plot landmark points
                # cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                # cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                # cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                # cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                # cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                # cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)

                current_state = self.exercise.get_state(int(knee_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self.exercise._update_state_sequence(current_state, self.state_tracker)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------

                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['SQUAT_COUNT'] += 1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['IMPROPER_SQUAT'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['IMPROPER_SQUAT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False


                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True


                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
                            self.state_tracker['state_seq'].count('s2') == 1:
                        self.state_tracker['DISPLAY_TEXT'][1] = True

                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                            self.state_tracker['state_seq'].count('s2') == 1:
                        self.state_tracker['LOWER_HIPS'] = True


                    elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][3] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                    if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time

                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        display_inactivity = True


                else:

                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # -------------------------------------------------------------------------------------------------------

                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                # if self.flip_frame:
                #     frame = cv2.flip(frame, 1)
                #     hip_text_coord_x = frame_width - hip_coord[0] + 10
                #     knee_text_coord_x = frame_width - knee_coord[0] + 15
                #     ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1

                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP,
                                            self.state_tracker['LOWER_HIPS'])

                if display_inactivity:
                    # cv2.putText(frame, 'Resetting COUNTERS due to inactivity!!!', (10, frame_height - 20), self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6,
                            self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1] + 10), self.font,
                            0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6,
                            self.COLORS['light_green'], 2, lineType=self.linetype)

                self.cv_elem.draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']),
                    pos=(int(frame_width * 0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                self.cv_elem.draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']),
                    pos=(int(frame_width * 0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                self.state_tracker['DISPLAY_TEXT'][
                    self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][
                    self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state




        else:

            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 25), self.font, 0.7, self.COLORS['blue'], 2)
                display_inactivity = True

            self.state_tracker['start_inactive_time'] = end_time

            self.cv_elem.draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']),
                pos=(int(frame_width * 0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            self.cv_elem.draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']),
                pos=(int(frame_width * 0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound






