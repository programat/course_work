# main_window_controller.py

import cv2
import sys
from PySide6.QtWidgets import QFileDialog, QWidget, QMessageBox
from src.controllers import opencv_controller

class MainWindowController:
    def __init__(self, window):
        self._window = window
        self.opencv_controller = opencv_controller.OpenCVController()

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, new):
        self._window = new

    def clicked_start(self):
        text = self.window.curls.text()
        style_sheet = self.window.curls.styleSheet()
        print(style_sheet, type(style_sheet))
        try:
            number = int(text)
            if 1 <= number <= 50:
                self.window.curls.setStyleSheet(style_sheet.rstrip('border-bottom: 1px solid red;'))
                print(f"Entered number {number} is correct!.")
            else:
                # setting a style with a red border to indicate invalid input
                self.window.curls.setStyleSheet(f"{style_sheet} border-bottom: 1px solid red;")
                QMessageBox.warning(self.window, "Warning", "Enter number from 1 to 50.")
        except ValueError:
            # setting a style with a red border to indicate incorrect input
            self.window.curls.setStyleSheet(f"{style_sheet} border-bottom: 1px solid red;")
            QMessageBox.warning(self.window, "Warning", "Enter correct number.")

    def chosen_stream(self):
        print(self.window.stream.currentText())

    def chosen_exercise(self):
        print(self.window.exercise.currentText())





