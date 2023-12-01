# pose_processor.py

class PoseProcessor:
    def __init__(self, angle_calculation_strategy):
        self.angle_calculation_strategy = angle_calculation_strategy

    def set_angle_calculation_strategy(self, angle_calculation_strategy):
        self.angle_calculation_strategy = angle_calculation_strategy

    def process_landmarks(self, landmarks):
        # Делегирование вычисления углов выбранной стратегии
        return self.angle_calculation_strategy.calculate_angles(landmarks)
