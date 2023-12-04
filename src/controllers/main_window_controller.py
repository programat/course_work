# main_window_controller.py

import cv2

class MainWindowController:
    def __init__(self):
        # Инициализация UI, создание окна и т.д.
        pass

    def display_info(self, info):
        # Отображение информации в окне
        cv2.putText(self.frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Your Window", self.frame)
        cv2.waitKey(1)
