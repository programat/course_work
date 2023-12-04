import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice
from PySide6.QtWidgets import QWidget
from src.controllers import main_window_controller
# from images import images_paths
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # self.controller = main_window_controller.MainWindowController(self)

        ui_file_name = "main.ui"
        ui_file = QFile(os.path.join(os.path.dirname(__file__), fr'../ui/{ui_file_name}'))
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        window = loader.load(ui_file)
        ui_file.close()
        if not window:
            print(loader.errorString())
            sys.exit(-1)

        # Ивент открытия
        # self.settings.clicked.connect(lambda : self.controller.clicked_settings())
        # self.dir.clicked.connect(lambda :self.controller.clicked_dir())
        # self.file.clicked.connect(lambda: self.controller.clicked_file())
        # self.start.clicked.connect(lambda :self.controller.clicked_start())
        # self.clearlogs.clicked.connect(lambda :self.controller.clicked_clear())

    def show(self):
        super().show()
        return self

    def closeEvent(self, QCloseEvent):
        self.controller.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())