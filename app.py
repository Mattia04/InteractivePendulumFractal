import simulation as sim
import numpy as np
import sys
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, qRgb
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout

PIXELS = 1024
ZOOM_FACTOR = 0.5  # 50% zoom per click

def time_to_rgb(value):
    # Blue (0, 0, 255) to Red (255, 0, 0) gradient
    if value < 0.25:
        return 4 * (0.25 - value) * 255, 0, 255
    if value < 0.5:
        return 0, 4 * (value - 0.25) * 255, 4 * (0.5 - value) * 255
    if value < 0.75:
        return 4 * (value - 0.5) * 255, 255, 0
    if value < 1:
        return 255, 4 * (1 - value) * 255, 0
    return 255, 255, 255


class ClickableImageLabel(QLabel):
    rangesUpdated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_level = 0
        self.current_ranges = {
            "q1min": -np.pi,
            "q1max": np.pi,
            "q2min": -np.pi,
            "q2max": np.pi,
        }
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.handle_zoom(event.pos(), zoom_in=True)
        elif event.button() == Qt.RightButton:
            self.handle_zoom(event.pos(), zoom_in=False)

    def handle_zoom(self, pos, zoom_in=True):
        # Calculate new ranges based on click position
        x = pos.x() / PIXELS
        y = pos.y() / PIXELS

        current_width = self.current_ranges["q1max"] - self.current_ranges["q1min"]
        current_height = self.current_ranges["q2max"] - self.current_ranges["q2min"]

        zoom_factor = ZOOM_FACTOR if zoom_in else 1 / ZOOM_FACTOR

        new_width = current_width * zoom_factor
        new_height = current_height * zoom_factor

        # Calculate new center based on click position
        center_q1 = self.current_ranges["q1min"] + x * current_width
        center_q2 = self.current_ranges["q2min"] + y * current_height

        # Update ranges
        self.current_ranges["q1min"] = center_q1 - new_width / 2
        self.current_ranges["q1max"] = center_q1 + new_width / 2
        self.current_ranges["q2min"] = center_q2 - new_height / 2
        self.current_ranges["q2max"] = center_q2 + new_height / 2

        # Update zoom level
        self.zoom_level += 1 if zoom_in else -1

        # Trigger update
        self.rangesUpdated.emit(self.current_ranges)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Pendulum Fractal Explorer")

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create image label
        self.image_label = ClickableImageLabel()
        self.image_label.rangesUpdated.connect(self.update_image)
        layout.addWidget(self.image_label)

        # Initial simulation
        self.update_image(self.image_label.current_ranges)

    def update_image(self, ranges):
        # Run simulation with current ranges
        data = sim.run_simulation(
            q1min=ranges["q1min"],
            q1max=ranges["q1max"],
            q2min=ranges["q2min"],
            q2max=ranges["q2max"],
        ).reshape((PIXELS, PIXELS))

        # Process data
        data[data < 0] = 100
        data = np.log(data + 1)
        data /= np.log(101)

        # Create QImage
        image = QImage(PIXELS, PIXELS, QImage.Format_RGB32)
        for x in range(PIXELS):
            for y in range(PIXELS):
                rgb = time_to_rgb(data[x, y])
                image.setPixel(x, y, qRgb(*map(int, rgb)))

        # Update display
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

#
# def rgb_value(value):
#     return np.array(time_to_rgb(value), dtype=np.uint8)
#
# def main():
#     # Call the simulation function
#     data = sim.run_simulation(
#         q1min=-3.1415, q1max=3.1415, q2min=-3.1415, q2max=3.1415
#     ).reshape((PIXELS, PIXELS))
#     data[data < 0] = 100
#
#     app = QApplication(sys.argv)
#
#     image = QImage(PIXELS, PIXELS, QImage.Format_RGB32)
#
#     data = np.log(data + 1)
#     data /= np.log(101)
#
#     for x in range(PIXELS):
#         for y in range(PIXELS):
#             image.setPixel(x, y, qRgb(*rgb_value(data[x, y])))
#
#     # Convert QImage to QPixmap.
#     pixmap = QPixmap.fromImage(image)
#
#     label = QLabel()
#     label.setPixmap(pixmap)
#     label.setFixedSize(pixmap.size())
#     label.show()
#
#     sys.exit(app.exec_())
#
#
# if __name__ == "__main__":
#     main()
