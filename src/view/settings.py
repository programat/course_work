import sys
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice
from src.controllers import settings_controller
import os

class Settings(QWidget):
    def __init__(self):
        super(Settings, self).__init__()
        self.controller = None
        self._settings_dict = settings_controller

    def create(self):
        ui_file_name = "settings.ui"
        ui_file = QFile(os.path.join(os.path.dirname(__file__), fr'../ui/{ui_file_name}'))
        if not ui_file.open(QFile.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        self.window_sets = loader.load(ui_file)
        ui_file.close()
        if not self.window_sets:
            print(loader.errorString())
            sys.exit(-1)

        self.controller = settings_controller.SettingsController(self.window_sets)

        # invents processed by controller
        self.window_sets.save_button.clicked.connect(lambda: self.controller.clicked_save())
        self.window_sets.folder_button.clicked.connect(lambda: self.controller.clicked_dir())
        self.window_sets.delete_button.clicked.connect(lambda: self.controller.clicked_delete())

        self.window_sets.setWindowTitle("Settings")
        return self.window_sets

    def get_settings(self):
        return self.controller.settings_dict

    def show(self):
        self.window_sets.show()
        return self.window_sets

    def closeEvent(self, QCloseEvent):
        # self.controller.close()
        pass