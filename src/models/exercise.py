# exercise.py

from abc import ABC, abstractmethod

class Exercise:
    def __init__(self):
        # name, parameters maybe
        # Инициализация упражнения
        pass

    @abstractmethod
    def _get_thresholds_beginner(self):
        pass

    @abstractmethod
    def _get_thresholds_pro(self):
        pass
    
    @abstractmethod
    def get_thresholds(self):
        pass

class SquatExercise(Exercise):

    def __init__(self, level=0):
        super().__init__()

        self.thresholds = None
        self._ANGLE_HIP_KNEE_VERT_BEGINNER = {
            'NORMAL': (0, 32),
            'TRANS': (35, 65),
            'PASS': (70, 95)
        }
        self.thresholds_beginner = {
            'HIP_KNEE_VERT': self._ANGLE_HIP_KNEE_VERT_BEGINNER,

            'HIP_THRESH': [10, 50],
            'ANKLE_THRESH': 45,
            'KNEE_THRESH': [50, 70, 95],

            'OFFSET_THRESH': 45.0,
            'INACTIVE_THRESH': 25.0,

            'CNT_FRAME_THRESH': 50
        }

        self._ANGLE_HIP_KNEE_VERT_PRO = {
            'NORMAL': (0, 32),
            'TRANS': (35, 65),
            'PASS': (80, 95)
        }
        self.thresholds_pro = {
            'HIP_KNEE_VERT': self._ANGLE_HIP_KNEE_VERT_PRO,

            'HIP_THRESH': [15, 50],
            'ANKLE_THRESH': 30,
            'KNEE_THRESH': [50, 80, 95],

            'OFFSET_THRESH': 45.0,
            'INACTIVE_THRESH': 15.0,

            'CNT_FRAME_THRESH': 50

        }
        
        self.set_level(level)

    def set_level(self, level=0):
        """setting the level of difficulty, 
        where 
        0 = beginner, 1 = pro
        """    
        if level:
            self.thresholds = self.thresholds_pro
        else:
            self.thresholds = self.thresholds_beginner

    def _get_thresholds_beginner(self):
        return self.thresholds_beginner

    def _get_thresholds_pro(self):
        return self.thresholds_pro
    
    def get_thresholds(self):
        return self.thresholds

    def get_state(self, knee_angle):

        knee = None

        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3

        return f's{knee}' if knee else None

    def _update_state_sequence(self, state, state_tracker):
        if state == 's2':
            if (('s3' not in state_tracker['state_seq']) and (state_tracker['state_seq'].count('s2')) == 0) or \
                    (('s3' in state_tracker['state_seq']) and (state_tracker['state_seq'].count('s2') == 1)):
                state_tracker['state_seq'].append(state)

        elif state == 's3':
            if (state not in state_tracker['state_seq']) and 's2' in state_tracker['state_seq']:
                state_tracker['state_seq'].append(state)

class DumbellExercise(Exercise):
    def __init__(self, level=0):
        super().__init__()

        self.thresholds = None
        self._ANGLE_HIP_KNEE_VERT_BEGINNER = {
            'NORMAL': (180, 110),
            'TRANS': (110, 70),
            'PASS': (70, 60)
        }
        self.thresholds_beginner = {
            'SHLDR_ELBOW_WRIST': self._ANGLE_HIP_KNEE_VERT_BEGINNER,

            'ELBOW_THRESH': [50, 60],
            'HAND_THRESH': [12, 20, 35],

            'OFFSET_THRESH': 45.0,
            'INACTIVE_THRESH': 25.0,
            'CNT_FRAME_THRESH': 50
        }

        self._ANGLE_HIP_KNEE_VERT_PRO = {
            'NORMAL': (180, 110),
            'TRANS': (110, 60),
            'PASS': (60, 50)
        }
        self.thresholds_pro = {
            'SHLDR_ELBOW_WRIST': self._ANGLE_HIP_KNEE_VERT_PRO,

            'ELBOW_THRESH': [50, 60],
            'HAND_THRESH': [12, 20, 35],

            'OFFSET_THRESH': 45.0,
            'INACTIVE_THRESH': 25.0,
            'CNT_FRAME_THRESH': 50
        }

        self.set_level(level)

    def set_level(self, level=0):
        """setting the level of difficulty,
        where
        0 = beginner, 1 = pro
        """
        if level:
            self.thresholds = self.thresholds_pro
        else:
            self.thresholds = self.thresholds_beginner

    def _get_thresholds_beginner(self):
        return self.thresholds_beginner

    def _get_thresholds_pro(self):
        return self.thresholds_pro

    def get_thresholds(self):
        return self.thresholds

    def get_state(self, elbow_angle):

        elbow = None

        if self.thresholds['SHLDR_ELBOW_WRIST']['NORMAL'][0] >= elbow_angle >= self.thresholds['SHLDR_ELBOW_WRIST']['NORMAL'][1]:
            elbow = 1
        elif self.thresholds['SHLDR_ELBOW_WRIST']['TRANS'][0] >= elbow_angle >= self.thresholds['SHLDR_ELBOW_WRIST']['TRANS'][1]:
            elbow = 2
        elif self.thresholds['SHLDR_ELBOW_WRIST']['PASS'][0] >= elbow_angle >= self.thresholds['SHLDR_ELBOW_WRIST']['PASS'][1]:
            elbow = 3

        return f's{elbow}' if elbow else None

    def _update_state_sequence(self, state, state_tracker):
        if state == 's2':
            if (('s3' not in state_tracker['state_seq']) and (state_tracker['state_seq'].count('s2')) == 0) or \
                    (('s3' in state_tracker['state_seq']) and (state_tracker['state_seq'].count('s2') == 1)):
                state_tracker['state_seq'].append(state)

        elif state == 's3':
            if (state not in state_tracker['state_seq']) and 's2' in state_tracker['state_seq']:
                state_tracker['state_seq'].append(state)