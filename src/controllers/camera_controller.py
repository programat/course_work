# camera_controller.py

class CameraController:
    def __init__(self, detection_strategy):
        self.detection_strategy = detection_strategy

    def set_detection_strategy(self, detection_strategy):
        self.detection_strategy = detection_strategy

    def process_image(self, frame):
        return self.detection_strategy.process_image(frame)
