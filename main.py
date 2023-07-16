import cv2
import numpy as np
import time
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, \
    QScrollArea, QListWidget, QDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFontMetrics, QIcon
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

class BlobData:
    def __init__(self):
        self.blob_positions = []
        self.fly_ids = {}

    def assign_fly_id(self, blob_id, fly_id):
        self.fly_ids[blob_id] = fly_id


class FlyData:
    def __init__(self, fly_id, centroid):
        self.fly_id = fly_id
        self.centroid = centroid
        self.mating = False
        self.mating_start_time = None

    def update_position(self, centroid):
        self.centroid = centroid


class MatingData:
    def __init__(self, fly1_id, fly2_id, mating_start_time):
        self.fly1_id = fly1_id
        self.fly2_id = fly2_id
        self.mating_start_time = mating_start_time


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    change_text_signal = pyqtSignal(int, str)
    change_elapsed_time_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.filename = None
        self.start_time = None
        self.rois = None
        self.roi_flies_mating_start_time = None
        self.roi_flies_last_mate_time = None
        self.roi_flies_verified_mating = None
        self.blob_data = []
        self.fly_data = []
        self.centroid_position_dialog = None
        self.roi_number = None

    def run(self):
        current_frame = 0
        if not self.filename:
            return

        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            print("Could not open video.")
            return

        min_roi_size = 30
        ret, first_frame = cap.read()
        if not ret:
            print("Cannot read video file.")
            return

        self.rois = self.define_roi(first_frame, min_roi_size)
        self.initialize_roi_data()

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 30
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        self.start_time = time.time()

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            frame_copy = frame.copy()

            for i, roi in enumerate(self.rois):
                roi_number = i + 1  # inside the loop over the ROIs
                roi_center, roi_radius = roi  # Retrieve roi_center and roi_radius
                cv2.circle(frame, roi_center, roi_radius, (0, 255, 0), 2)
                # Add ROI number
                text_position = (roi[0][0] - 10, roi[0][1] - roi[1] - 10)
                cv2.putText(frame_copy, str(roi_number), text_position, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

            # Rest of the code...

            elapsed_time = time.time() - self.start_time
            self.change_elapsed_time_signal.emit(self.format_time(elapsed_time))

            gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            keypoints = detector.detect(gray)

            for i, roi in enumerate(self.rois):
                roi_center, roi_radius = roi

                blobs_in_roi = [kp for kp in keypoints if abs(kp.pt[0] - roi_center[0]) <= roi_radius and abs(
                    kp.pt[1] - roi_center[1]) <= roi_radius]
                blobs_in_roi.sort(key=lambda kp: kp.size, reverse=True)
                blobs_in_roi = blobs_in_roi[:2]

                color = (0, 0, 255)

                if len(blobs_in_roi) == 1:
                    now = time.time()
                    if self.roi_flies_mating_start_time[i] is None or (
                            self.roi_flies_last_mate_time[i] is not None and now - self.roi_flies_last_mate_time[
                        i] < 2):
                        self.roi_flies_mating_start_time[i] = now if self.roi_flies_mating_start_time[i] is None else \
                            self.roi_flies_mating_start_time[i]
                        mating_duration = now - self.roi_flies_mating_start_time[i]
                        self.roi_flies_last_mate_time[i] = now

                        if self.roi_flies_verified_mating[i]:
                            if mating_duration > 120:
                                color = (255, 0, 0)
                                mating_time = self.format_time(mating_duration)
                                mating_text = f"Mating for {mating_time} (verified mating)"
                                self.change_text_signal.emit(i + 1, mating_text)
                            else:
                                color = (0, 255, 255)
                                mating_time = self.format_time(mating_duration)
                                self.change_text_signal.emit(i + 1, f"Mating for {mating_time}")
                        else:
                            if mating_duration > 120:
                                self.roi_flies_verified_mating[i] = True
                                color = (255, 0, 0)
                                mating_time = self.format_time(mating_duration)
                                mating_text = f"Mating for {mating_time} (verified mating)"
                                self.change_text_signal.emit(i + 1, mating_text)
                            else:
                                color = (0, 255, 255)
                                mating_time = self.format_time(mating_duration)
                                self.change_text_signal.emit(i + 1, f"Mating for {mating_time}")
                    else:
                        self.roi_flies_mating_start_time[i] = None
                        self.roi_flies_last_mate_time[i] = None

                    fly_size = 10  # Adjust the size as desired
                    fly_center = (int(blobs_in_roi[0].pt[0]), int(blobs_in_roi[0].pt[1]))
                    fly_thickness = 2  # Adjust the thickness as desired

                    # Get the fly ID based on the blob ID
                    blob_id = blobs_in_roi[0].class_id
                    blob_data = self.blob_data[i]
                    fly_data = self.get_fly_data(blob_id, blob_data)

                    if fly_data is None:
                        # Assign a new ID for the fly
                        fly_id = len(self.fly_data) + 1
                        fly_data = FlyData(fly_id, fly_center)
                        self.fly_data.append(fly_data)
                        blob_data.assign_fly_id(blob_id, fly_id)
                    else:
                        fly_data.update_position(fly_center)

                    fly_label = "F" if fly_data.fly_id == 1 else "M"
                    cv2.circle(frame, fly_center, fly_size, color, fly_thickness)
                    cv2.putText(frame, fly_label, (fly_center[0] - 10, fly_center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Add the centroid position to the blob_data structure
                    blob_data.blob_positions.append(fly_center)

                else:
                    self.roi_flies_mating_start_time[i] = None
                    self.roi_flies_last_mate_time[i] = None

                    fly_size = 10  # Adjust the size as desired
                    fly_thickness = 2  # Adjust the thickness as desired
                    for j, kp in enumerate(blobs_in_roi):
                        fly_center = (int(kp.pt[0]), int(kp.pt[1]))

                        # Get the fly ID based on the blob ID
                        blob_id = kp.class_id
                        blob_data = self.blob_data[i]
                        fly_data = self.get_fly_data(blob_id, blob_data)

                        if fly_data is None:
                            # Assign a new ID for the fly
                            fly_id = len(self.fly_data) + 1
                            fly_data = FlyData(fly_id, fly_center)
                            self.fly_data.append(fly_data)
                            blob_data.assign_fly_id(blob_id, fly_id)
                        else:
                            fly_data.update_position(fly_center)

                        fly_label = "F" if fly_data.fly_id == 1 else "M"
                        cv2.circle(frame, fly_center, fly_size, color, fly_thickness)
                        cv2.putText(frame, fly_label, (fly_center[0] - 10, fly_center[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.detect_mating_events()

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)

            painter = QPainter(p)
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)

            for i, roi in enumerate(self.rois):
                roi_center, roi_radius = roi
                roi_label = f"ROI {i + 1}"
                metrics = QFontMetrics(font)
                text_width = metrics.horizontalAdvance(roi_label)
                text_height = metrics.height()

                x = roi_center[0] - text_width // 2
                y = roi_center[1] + roi_radius + text_height + 5
                painter.setPen(Qt.white)
                painter.drawText(x, y, roi_label)

                roi_number = str(i + 1)
                number_width = metrics.horizontalAdvance(roi_number)
                number_x = roi_center[0] - number_width // 2
                number_y = roi_center[1] + roi_radius + text_height + 25
                painter.drawText(number_x, number_y, roi_number)

            painter.end()

            self.change_pixmap_signal.emit(p)
            cv2.waitKey(1)

        self.save_centroid_positions()

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._run_flag = False

    def set_filename(self, filename):
        self.filename = filename

    def define_roi(self, frame, min_roi_size):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        rois = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circles = sorted(circles, key=lambda x: x[0])
            for (x, y, r) in circles:
                if r >= min_roi_size:
                    roi = [(x, y), r]
                    is_overlapping = False
                    for existing_roi in rois:
                        existing_center, existing_radius = existing_roi
                        dist = np.sqrt((x - existing_center[0]) ** 2 + (y - existing_center[1]) ** 2)
                        if dist < existing_radius or dist < r:
                            is_overlapping = True
                            break
                    if not is_overlapping and len(rois) < 6:
                        rois.append(roi)
                        self.blob_data.append(BlobData())
        return rois

    def initialize_roi_data(self):
        self.roi_flies_mating_start_time = [None] * len(self.rois)
        self.roi_flies_last_mate_time = [None] * len(self.rois)
        self.roi_flies_verified_mating = [False] * len(self.rois)

    @staticmethod
    def format_time(duration):
        milliseconds = int(duration * 1000) % 1000
        seconds = int(duration) % 60
        minutes = int(duration // 60) % 60
        hours = int(duration // 3600)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def save_centroid_positions(self):
        df = pd.DataFrame()
        for i, blob_data in enumerate(self.blob_data):
            roi_label = f"ROI {i + 1}"
            centroids = blob_data.blob_positions
            x_values = [centroid[0] for centroid in centroids]
            y_values = [centroid[1] for centroid in centroids]
            df[roi_label + "_X"] = x_values
            df[roi_label + "_Y"] = y_values

        filename = "centroid_positions.xlsx"
        df.to_excel(filename, index=False)
        print(f"Centroid positions saved to {filename}.")

    def get_fly_data(self, blob_id, blob_data):
        for fly_data in self.fly_data:
            if fly_data.fly_id == blob_data.fly_ids.get(blob_id):
                return fly_data
        return None

    def detect_mating_events(self):
        mating_distance_threshold = 4

        for i, roi in enumerate(self.rois):
            blob_data = self.blob_data[i]
            centroids = blob_data.blob_positions

            for j, centroid1 in enumerate(centroids):
                fly_data1 = self.get_fly_data(j + 1, blob_data)
                if fly_data1 is None:
                    continue

                for k, centroid2 in enumerate(centroids[j + 1:], start=j + 1):
                    fly_data2 = self.get_fly_data(k + 1, blob_data)
                    if fly_data2 is None:
                        continue

                    distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
                    if distance <= mating_distance_threshold:
                        if not fly_data1.mating and not fly_data2.mating:
                            fly_data1.mating = True
                            fly_data2.mating = True
                            fly_data1.mating_start_time = time.time()
                            fly_data2.mating_start_time = time.time()
                            self.change_text_signal.emit(i + 1, "Mating")

                    if fly_data1.mating and time.time() - fly_data1.mating_start_time > 120:
                        fly_data1.mating = False
                        fly_data1.mating_start_time = None
                        self.change_text_signal.emit(i + 1, "")
                        if self.roi_flies_verified_mating[i]:
                            self.roi_flies_verified_mating[i] = False

                    if fly_data2.mating and time.time() - fly_data2.mating_start_time > 120:
                        fly_data2.mating = False
                        fly_data2.mating_start_time = None
                        self.change_text_signal.emit(i + 1, "")
                        if self.roi_flies_verified_mating[i]:
                            self.roi_flies_verified_mating[i] = False

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fly Mating Detection")
        self.resize(900, 600)

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.change_text_signal.connect(self.update_text)
        self.video_thread.change_elapsed_time_signal.connect(self.update_elapsed_time)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.text_labels = []
        for i in range(6):
            label = QLabel(self)
            self.text_labels.append(label)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_video)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)

        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)

        self.roi_dialog = QDialog(self)
        self.roi_dialog.setWindowTitle("Select ROIs")
        self.roi_dialog.setLayout(QVBoxLayout())

        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QListWidget.MultiSelection)
        self.roi_list.itemSelectionChanged.connect(self.update_roi_selection)

        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.clicked.connect(self.add_roi)

        self.remove_roi_button = QPushButton("Remove ROI")
        self.remove_roi_button.clicked.connect(self.remove_roi)

        self.done_roi_button = QPushButton("Done")
        self.done_roi_button.clicked.connect(self.close_roi_dialog)

        self.centroid_button = QPushButton("Show Centroid Position")
        self.centroid_button.clicked.connect(self.show_centroid_position)

        self.roi_dialog.layout().addWidget(self.roi_list)
        self.roi_dialog.layout().addWidget(self.add_roi_button)
        self.roi_dialog.layout().addWidget(self.remove_roi_button)
        self.roi_dialog.layout().addWidget(self.done_roi_button)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)

        text_layout = QHBoxLayout()
        for label in self.text_labels:
            text_layout.addWidget(label)
        self.layout.addLayout(text_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.centroid_button)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

    def start_video(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.select_button.setEnabled(False)
        self.centroid_button.setEnabled(True)

        self.video_thread.start()

    def stop_video(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.select_button.setEnabled(True)
        self.centroid_button.setEnabled(False)

        self.video_thread.stop()

    def select_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video")
        if filename:
            self.video_thread.set_filename(filename)
            self.show_roi_dialog()

    def show_roi_dialog(self):
        self.roi_list.clear()
        self.roi_dialog.show()

    def update_roi_selection(self):
        selected_items = self.roi_list.selectedItems()
        selected_indices = [self.roi_list.row(item) for item in selected_items]
        self.video_thread.update_selected_rois(selected_indices)

    def add_roi(self):
        num_items = self.roi_list.count()
        if num_items < 6:
            self.roi_list.addItem(f"ROI {num_items + 1}")

    def remove_roi(self):
        selected_items = self.roi_list.selectedItems()
        for item in selected_items:
            self.roi_list.takeItem(self.roi_list.row(item))

    def close_roi_dialog(self):
        self.roi_dialog.close()

    def show_centroid_position(self):
        centroid_dialog = QDialog(self)
        centroid_dialog.setWindowTitle("Centroid Positions")
        centroid_dialog.setLayout(QVBoxLayout())

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        centroid_widget = QWidget()
        centroid_widget.setLayout(QVBoxLayout())

        for i, blob_data in enumerate(self.video_thread.blob_data):
            roi_label = f"ROI {i + 1}"
            centroids = blob_data.blob_positions

            centroid_label = QLabel(f"{roi_label} Centroid Positions:")
            centroid_widget.layout().addWidget(centroid_label)

            for j, centroid in enumerate(centroids):
                centroid_position = QLabel(f"Centroid {j + 1}: ({centroid[0]}, {centroid[1]})")
                centroid_widget.layout().addWidget(centroid_position)

            centroid_widget.layout().addSpacing(10)

        scroll_area.setWidget(centroid_widget)
        centroid_dialog.layout().addWidget(scroll_area)

        centroid_dialog.exec_()

    @pyqtSlot(QImage)
    def update_image(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(int, str)
    def update_text(self, index, text):
        if index >= 1 and index <= 6:
            label = self.text_labels[index - 1]
            label.setText(f"ROI {index}: {text}")

    @pyqtSlot(str)
    def update_elapsed_time(self, time_str):
        roi_number = self.video_thread.roi_number  # assuming that roi_number is a property of the VideoThread class
        self.setWindowTitle(f"Fly Mating Detection - ROI: {roi_number} - Elapsed Time: {time_str}")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
