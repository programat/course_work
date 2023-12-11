# dumbbell_processor.py

from src.strategies.pose_processor.pose_processor import PoseProcessor
import time
import cv2
import numpy as np
from src.strategies import angle_calculation_strategy as acs
from src.strategies import detection_strategy as dc
from src.models import exercise as exr


class DumbbellProcessor(PoseProcessor):
    def __init__(self, detection_strategy: dc.DetectionStrategy,
                 angle_calculation_strategy: acs.AngleCalculationStrategy, level=0):
        super().__init__(detection_strategy, angle_calculation_strategy, level)

        self.exercise = exr.DumbellExercise(level)
        self.thresholds = self.exercise.get_thresholds()

        self.state_tracker = {
            'state_seq': [],

            'NEAR_HAND': False,

            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,
            'INACTIVE_TIME_START': 0.0,

            # 0 -> LOWER YOUR WRIST, 1 -> HIGHER YOUR WRIST, 2 -> KEEP YOUR HAND NEAR THE BODY
            'DISPLAY_TEXT': np.full((3,), False),
            'COUNT_FRAMES': np.zeros((3,), dtype=np.int64),

            'INCORRECT_POSTURE': False,

            'prev_state': None,
            'curr_state': None,

            'CURLS': 0,
            'BAD_CURLS': 0
        }

        self.FEEDBACK_ID_MAP = {
            0: ('LOWER YOUR WRIST', 125, self.COLORS['purple']),
            1: ('HIGHER YOUR WRIST', 125, self.COLORS['pink']),
            2: ('KEEP YOUR HAND NEAR THE BODY', 60, (255, 80, 80))
        }

    def calculate_angle(self, p1, p2, ref_pt=np.array([0, 0])):
        return self.angle_calculation.calculate_angle(p1, p2, ref_pt)

    def _show_feedback(self, frame, c_frame, dict_maps, near_hand_disp):
        if near_hand_disp:
            self.cv_elem.draw_text(
                frame,
                'HAND TOO FAR FROM BODY',
                pos=(int(frame.shape[1] * 0.06), 60),
                text_color=(0, 0, 0),
                font_scale=1,
                font_thickness=3,
                text_color_bg=(255, 255, 0),
                increased_size=3
            )
            c_frame[2] = False

        # if c_frame[0]: c_frame[1] = False
        for idx in np.where(c_frame)[0]:
            self.cv_elem.draw_text(
                frame,
                dict_maps[idx][0],
                pos=(int(frame.shape[1] * 0.06), dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=1,
                font_thickness=3,
                text_color_bg=dict_maps[idx][2],
                increased_size=3
            )
        return frame

    def draw_landmark_line(self, frame, p1, p2, color, thickness):
        if not np.array_equal(p1, [0, 0]) and not np.array_equal(p2, [0, 0]):
            cv2.line(frame, p1, p2, color, thickness, lineType=self.linetype)
        return frame

    def process(self, frame: np.array, curls=None):
        frame_height, frame_width, _ = frame.shape

        keypoints = self.detector.get_coordinates()

        if len(keypoints):
            nose_coord = self.detector.get_landmark_coordinates('nose')
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord = \
                self.detector.get_landmark_coordinates('left')
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord = \
                self.detector.get_landmark_coordinates('right')

            offset_angle = self.calculate_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']:
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker[
                    'start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['CURLS'] = 0
                    self.state_tracker['BAD_CURLS'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                if display_inactivity or (time.time() - self.state_tracker['INACTIVE_TIME_START']) <= 3:
                    cv2.putText(frame, 'Resetting CURLS due to inactivity!', (10, 90),
                                self.font, 0.5, self.COLORS['red'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                    if not (time.time() - self.state_tracker['INACTIVE_TIME_START']) <= 3:
                        self.state_tracker['INACTIVE_TIME_START'] = time.time()

                self.cv_elem.draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CURLS']),
                    pos=(int(frame_width * 0.06), int(frame_height - 80)),
                    text_color=(255, 255, 230),
                    font_scale=1,
                    font_thickness=3,
                    text_color_bg=(10, 228, 72),
                    increased_size=3
                )

                self.cv_elem.draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['BAD_CURLS']),
                    pos=(int(frame_width * 0.78), int(frame_height - 80)),
                    text_color=(255, 255, 230),
                    font_scale=1,
                    font_thickness=3,
                    text_color_bg=(254, 197, 251),
                    increased_size=3
                )

                if curls is not None:
                    self.cv_elem.draw_text(
                        frame,
                        "CURLS: " + str(self.state_tracker['CURLS'] + self.state_tracker['BAD_CURLS']) + '/' + str(
                            curls),
                        pos=(int(frame_width / 2.1), frame_height - 30),
                        text_color=(255, 255, 230),
                        font_scale=0.5,
                        text_color_bg=self.COLORS['black']
                    )

                self.cv_elem.draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, 60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                self.cv_elem.draw_text(
                    frame,
                    'OFFSET ANGLE: ' + str(offset_angle),
                    pos=(30, 30),
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

                dist_l = abs(left_elbow_coord[1] - left_shldr_coord[1])
                dist_r = abs(right_elbow_coord[1] - right_shldr_coord)[1]

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None

                if nose_coord[0] <= left_shldr_coord[0]:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord

                    multiplier = -1

                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord

                    multiplier = 1

                # --- Calculation angles ----

                elbow_angle = self.calculate_angle(shldr_coord, wrist_coord, elbow_coord)
                cv2.ellipse(frame, elbow_coord, (30, 30),
                            angle=elbow_angle, startAngle=0, endAngle=0,
                            color=self.COLORS['white'], thickness=3, lineType=self.linetype)

                shldr_angle = abs(180 - self.calculate_angle(elbow_coord, np.array([shldr_coord[0], 0]), shldr_coord))
                if shldr_coord[0] >= elbow_coord[0]:
                    cv2.ellipse(frame, shldr_coord, (30, 30),
                                angle=0, startAngle=90 - multiplier * shldr_angle, endAngle=90,
                                color=self.COLORS['white'], thickness=3, lineType=self.linetype)
                else:
                    cv2.ellipse(frame, shldr_coord, (30, 30),
                                angle=0, startAngle=90 + multiplier * shldr_angle, endAngle=90,
                                color=self.COLORS['white'], thickness=3, lineType=self.linetype)

                self.cv_elem.draw_dotted_line(frame, shldr_coord, start=shldr_coord[1] - 20, end=shldr_coord[1] + 50,
                                              line_color=self.COLORS['purple'])

                if not self.detector.is_plotted:
                    # plotting landmarks
                    self.draw_landmark_line(frame, shldr_coord, elbow_coord, self.COLORS['pink'], 4)
                    self.draw_landmark_line(frame, wrist_coord, elbow_coord, self.COLORS['pink'], 4)

                    # plotting edges of landmarks
                    cv2.circle(frame, shldr_coord, 7, self.COLORS['white'], -1, lineType=self.linetype)
                    cv2.circle(frame, elbow_coord, 7, self.COLORS['white'], -1, lineType=self.linetype)
                    cv2.circle(frame, wrist_coord, 7, self.COLORS['white'], -1, lineType=self.linetype)

                current_state = self.exercise.get_state(int(elbow_angle))
                self.state_tracker['curr_state'] = current_state
                self.exercise._update_state_sequence(current_state, self.state_tracker)

                # --- Computing parts of automata

                # print('\r', self.state_tracker['state_seq'], current_state, self.state_tracker['DISPLAY_TEXT'], self.state_tracker['COUNT_FRAMES'], self.state_tracker['NEAR_HAND'], end='')

                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['CURLS'] += 1
                        play_sound = str(self.state_tracker['CURLS'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['BAD_CURLS'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['BAD_CURLS'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                    # --- End of computing

                # --- Perform feedback

                else:
                    if elbow_angle < self.thresholds['ELBOW_THRESH'][0]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True

                    elif elbow_angle > self.thresholds['ELBOW_THRESH'][1] and \
                            self.state_tracker['state_seq'].count('s2') == 1:
                        self.state_tracker['DISPLAY_TEXT'][1] = True

                    if self.thresholds['HAND_THRESH'][0] < shldr_angle < self.thresholds['HAND_THRESH'][1] and \
                            self.state_tracker['state_seq'].count('s2') == 1:
                        self.state_tracker['DISPLAY_TEXT'][2] = True

                    elif shldr_angle > self.thresholds['HAND_THRESH'][2]:
                        self.state_tracker['NEAR_HAND'] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                # --- Inactivity computing

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time

                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['CURLS'] = 0
                        self.state_tracker['BAD_CURLS'] = 0
                        display_inactivity = True

                else:
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # ---

                elbow_text_coord_x = elbow_coord[0] + 10
                shldr_text_coord_x = shldr_coord[0] + 15

                if current_state == 's1':
                    self.state_tracker['NEAR_HAND'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1

                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'],
                                            self.FEEDBACK_ID_MAP, self.state_tracker['NEAR_HAND'])

                if display_inactivity or (time.time() - self.state_tracker['INACTIVE_TIME_START']) <= 3:
                    cv2.putText(frame, 'Resetting CURLS due to inactivity!', (10, 90),
                                self.font, 0.5, self.COLORS['red'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                    if not (time.time() - self.state_tracker['INACTIVE_TIME_START']) <= 3:
                        self.state_tracker['INACTIVE_TIME_START'] = time.time()

                cv2.putText(frame, str(int(elbow_angle)), (elbow_text_coord_x, elbow_coord[1]), self.font, 0.6,
                            self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(shldr_angle)), (shldr_text_coord_x, shldr_coord[1] + 10),
                            self.font,
                            0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

                self.cv_elem.draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CURLS']),
                    pos=(int(frame_width * 0.06), int(frame_height - 80)),
                    text_color=(255, 255, 230),
                    font_scale=1,
                    font_thickness=3,
                    text_color_bg=(10, 228, 72),
                    increased_size=3
                )

                self.cv_elem.draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['BAD_CURLS']),
                    pos=(int(frame_width * 0.78), int(frame_height - 80)),
                    text_color=(255, 255, 230),
                    font_scale=1,
                    font_thickness=3,
                    text_color_bg=(254, 197, 251),
                    increased_size=3
                )

                if curls is not None:
                    self.cv_elem.draw_text(
                        frame,
                        "CURLS: " + str(self.state_tracker['CURLS'] + self.state_tracker['BAD_CURLS']) + '/' + str(
                            curls),
                        pos=(int(frame_width / 2.1), frame_height - 30),
                        text_color=(255, 255, 230),
                        font_scale=0.5,
                        text_color_bg=self.COLORS['black']
                    )

                self.state_tracker['DISPLAY_TEXT'][
                    self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][
                    self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state


        # --- if len(keypoints)

        else:
            # if self.flip_frame:
            #     frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH'] or (
                    time.time() - self.state_tracker['INACTIVE_TIME_START']) <= 3:
                self.state_tracker['CURLS'] = 0
                self.state_tracker['BAD_CURLS'] = 0
                cv2.putText(frame, 'Resetting CURLS due to inactivity!!!', (10, frame_height - 25), self.font, 0.7,
                            self.COLORS['red'], 2)
                display_inactivity = True
                if not (time.time() - self.state_tracker['INACTIVE_TIME_START']) <= 3:
                    self.state_tracker['INACTIVE_TIME_START'] = time.time()

            self.state_tracker['start_inactive_time'] = end_time

            self.cv_elem.draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['CURLS']),
                pos=(int(frame_width * 0.06), int(frame_height - 80)),
                text_color=(255, 255, 230),
                font_scale=1,
                font_thickness=3,
                text_color_bg=(10, 228, 72),
                increased_size=3
            )

            self.cv_elem.draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['BAD_CURLS']),
                pos=(int(frame_width * 0.78), int(frame_height - 80)),
                text_color=(255, 255, 230),
                font_scale=1,
                font_thickness=3,
                text_color_bg=(254, 197, 251),
                increased_size=3
            )

            if curls is not None:
                self.cv_elem.draw_text(
                    frame,
                    "CURLS: " + str(self.state_tracker['CURLS'] + self.state_tracker['BAD_CURLS']) + '/' + str(
                        curls),
                    pos=(int(frame_width / 2.1), frame_height - 30),
                    text_color=(255, 255, 230),
                    font_scale=0.5,
                    font_thickness=0.5,
                    text_color_bg=self.COLORS['black']
                )

            if display_inactivity or (time.time() - self.state_tracker['INACTIVE_TIME_START']) <= 3:
                cv2.putText(frame, 'Resetting CURLS due to inactivity!', (10, 90),
                            self.font, 0.5, self.COLORS['red'], 2, lineType=self.linetype)
                play_sound = 'reset_counters'
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                if not (time.time() - self.state_tracker['INACTIVE_TIME_START']) <= 3:
                    self.state_tracker['INACTIVE_TIME_START'] = time.time()

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame
