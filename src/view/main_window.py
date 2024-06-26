import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice
from src.controllers import main_window_controller
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.controller = None


    def create(self):
        ui_file_name = "main-new.ui"
        ui_file = QFile(os.path.join(os.path.dirname(__file__), fr'../ui/{ui_file_name}'))
        if not ui_file.open(QFile.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        if not self.window:
            print(loader.errorString())
            sys.exit(-1)

        self.controller = main_window_controller.MainWindowController(self.window)

        # invents processed by controller
        self.window.settings.clicked.connect(lambda: self.controller.clicked_settings())
        self.window.stream.currentIndexChanged.connect(lambda: self.controller.chosen_stream())
        self.window.exercise.currentIndexChanged.connect(lambda: self.controller.chosen_exercise())
        self.window.start_button.clicked.connect(lambda: self.controller.clicked_start())
        self.window.folder_button.clicked.connect(lambda: self.controller.clicked_dir())

        self.window.setWindowTitle("AI Trainer")
        return self.window

    def show(self):
        self.window.show()
        return self.window

    def closeEvent(self, QCloseEvent):
        self.controller.close()
        pass

if __name__ == '__main__':
    app = QApplication([])
    w = MainWindow().create()
    w.show()
    sys.exit(app.exec())