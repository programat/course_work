# main_window_controller.py

from PySide6.QtWidgets import QFileDialog, QWidget, QMessageBox
from src.controllers import opencv_controller
from src.strategies import detection_strategy
from src.strategies import pose_processor_strategy
from src.strategies import angle_calculation_strategy

class MainWindowController:
    def __init__(self, window):
        self._window = window
        self.opencv_controller = None

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, new):
        self._window = new
    #
    def clicked_start(self):
        text = self.window.curls.text()
        style_sheet = self.window.curls.styleSheet()
        try:
            number = int(text)
            if 1 <= number <= 50:
                self.window.curls.setStyleSheet(style_sheet.rstrip('border-bottom: 1px solid red;'))
                print(f"Entered number {number} is correct!. Starting...")
                detector = detection_strategy.YOLOStrategy().create_model()
                angle = angle_calculation_strategy.Angle2DCalculation()
                self.opencv_controller = opencv_controller.OpenCVController(detector, angle, self.chosen_exercise())
                self.opencv_controller.setup(stream=self.chosen_stream(), level=self.get_level())
                self.opencv_controller.process(show_fps=True, curls=number)
            else:
                # setting a style with a red border to indicate invalid input
                self.window.curls.setStyleSheet(f"{style_sheet} border-bottom: 1px solid red;")
                QMessageBox.warning(self.window, "Warning", "Enter number from 1 to 50.")
        except ValueError:
            # setting a style with a red border to indicate incorrect input
            self.window.curls.setStyleSheet(f"{style_sheet} border-bottom: 1px solid red;")
            QMessageBox.warning(self.window, "Warning", "Enter correct number.")

    def chosen_stream(self):
        if self.window.stream.currentText() == 'Vid': return 0
        return 1

    def chosen_exercise(self):
        return self.window.exercise.currentText()

    def get_level(self):
        if self.window.radioButton.pressed: return 0
        return 1



