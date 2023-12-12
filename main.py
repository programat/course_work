
from src.view.main_window import MainWindow
from PySide6.QtWidgets import QApplication
import sys

if __name__ == '__main__':

    app = QApplication([])
    w = MainWindow().create()
    w.show()
    sys.exit(app.exec())