# settings_controller.py

from PySide6.QtWidgets import QMessageBox, QFileDialog

class SettingsController:
    def __init__(self, window):
        self._settings_dict = {}
        self._window = window
        self.opencv_controller = None

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, new):
        self._window = new

    def get_path(self):
        print(self.window.file.text())
        return self.window.file.text()

    def clicked_dir(self):
        dialog = QFileDialog()
        fname = dialog.getOpenFileName(self.window, 'Open file', '/home')[0]
        self.window.file.setText(fname)

    def choose_param(self):
        return (self.window.cb1.isChecked(), self.window.cb2.isChecked())

    def set_param(self, p1=False, p2=False):
        self.window.cb1.setTristate(p1)
        self.window.cb2.setTristate(p2)
        return p1, p2


    def chosen_size(self):
        return self.window.imgsz.text()

    def chosen_conf(self):
        return self.window.conf.text()

    def chosen_iou(self):
        return self.window.iou.text()

    def clicked_save(self):
        self.fps, self.plot = self.choose_param()
        self.settings_dict = {
            'path': self.get_path(),
            'imgsz': int(self.chosen_size()),
            'conf': float(self.chosen_conf()),
            'iou': float(self.chosen_iou()),
            'fps': int(self.fps),
            'plot': bool(self.plot)
        }
        print(self.settings_dict)

        self.window.close()

    def clicked_delete(self):
        self.fps, self.plot = self.set_param()
        self.window.file.setText('')
        self.window.imgsz.setText(str(320))
        self.window.conf.setText(str(0.25))
        self.window.iou.setText(str(0.7))

        self.settings_dict = {
            'path': self.get_path(),
            'imgsz': int(self.chosen_size()),
            'conf': float(self.chosen_conf()),
            'iou': float(self.chosen_iou()),
            'fps': int(self.fps),
            'plot': bool(self.plot)
        }
        print(self.settings_dict)

    @property
    def settings_dict(self):
        return self._settings_dict

    @settings_dict.setter
    def settings_dict(self, value):
        self._settings_dict = value






