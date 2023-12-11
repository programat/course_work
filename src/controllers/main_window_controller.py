# main_window_controller.py

from PySide6.QtWidgets import QMessageBox, QFileDialog
from src.view import settings
from src.controllers import opencv_controller
from src.strategies import detection_strategy
from src.strategies import angle_calculation_strategy

class MainWindowController:
    def __init__(self, window):
        self._window = window
        self.opencv_controller = None
        self.sets = settings.Settings()
        self.sets.create()

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, new):
        self._window = new

    def clicked_settings(self):
        self.sets.show()

    def clicked_start(self):
        text = self.window.curls.text()
        style_sheet = self.window.curls.styleSheet()
        try:
            number = int(text)
            if 1 <= number <= 50:
                self.window.curls.setStyleSheet(style_sheet.rstrip('border-bottom: 1px solid red;'))
                print(f"Entered number {number} is correct!. Starting...")

                settings_dict = self.sets.get_settings()

                if len(settings_dict) > 0:
                    if settings_dict['path'] != '':
                        self.detector = detection_strategy.YOLOStrategy(imgsz=settings_dict['imgsz'], weights_path=settings_dict['path'], conf=settings_dict['conf'], iou=settings_dict['iou']).create_model()
                    else:
                        self.detector = detection_strategy.YOLOStrategy(imgsz=settings_dict['imgsz'], conf=settings_dict['conf'], iou=settings_dict['iou']).create_model()
                else:
                    self.detector = detection_strategy.YOLOStrategy().create_model()

                angle = angle_calculation_strategy.Angle2DCalculation()
                self.opencv_controller = opencv_controller.OpenCVController(self.detector, angle, self.chosen_exercise())
                if self.chosen_stream():
                    self.opencv_controller.setup(stream=self.chosen_stream(), level=self.get_level())
                else:
                    self.opencv_controller.setup(stream=self.chosen_stream(), level=self.get_level(), video_path=self.get_path())
                if len(settings_dict) > 0:
                    self.opencv_controller.process(show_fps=settings_dict['fps'], curls=number, plot=settings_dict['plot'])
                else:
                    self.opencv_controller.process(show_fps=False, curls=number, plot=False)
            else:
                # setting a style with a red border to indicate invalid input
                self.window.curls.setStyleSheet(f"{style_sheet} border-bottom: 1px solid red;")
                QMessageBox.warning(self.window, "Warning", "Enter number from 1 to 50.")
        except ValueError as v:
            print(v)
            # setting a style with a red border to indicate incorrect input
            self.window.curls.setStyleSheet(f"{style_sheet} border-bottom: 1px solid red;")
            QMessageBox.warning(self.window, "Warning", "Enter correct number.")

    def get_path(self):
        return self.window.file.text()

    def clicked_dir(self):
        dialog = QFileDialog()
        fname = dialog.getOpenFileName(self.window, 'Open file', '/home')[0]
        self.window.file.setText(fname)

    def chosen_stream(self):
        if self.window.stream.currentText() == 'Vid':
            self.window.file_label.setEnabled(True)
            self.window.file.setEnabled(True)
            self.window.folder_button.setEnabled(True)
            return 0
        else:
            self.window.file_label.setEnabled(False)
            self.window.file.setEnabled(False)
            self.window.folder_button.setEnabled(False)
            return 1

    def chosen_exercise(self):
        return self.window.exercise.currentText()

    def get_level(self):
        if self.window.radioButton.pressed: return 0
        return 1



