import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QLineEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import pandas as pd

class VideoProcessingThread(QThread):
    finished = pyqtSignal()
    frame_processed = pyqtSignal(np.ndarray, dict)
    frame_info = pyqtSignal(int, float)
    verified_mating_start_times = pyqtSignal(dict)
    fly_info = pyqtSignal(int, object)
    binary_image_processed = pyqtSignal(np.ndarray)

    def __init__(self, video_path, initial_contours, fps):
        super().__init__()
        self.video_path = video_path
        self.initial_contours = initial_contours
        self.is_running = False
        self.roi_ids = {}  # Dictionary to store ROI IDs
        self.mating_start_times = {}  # Dictionary to store mating start times for each ROI
        self.mating_durations = {}  # Dictionary to store mating durations for each ROI
        self.fps = fps
        self.mating_start_frames = {}  # Dictionary to store mating start frames for each ROI
        self.mating_grace_frames = {}  # Dictionary to store grace frames for each ROI
        self.mating_start_times_df = pd.DataFrame(columns=['ROI', 'Start Times'])  # Create an empty DataFrame to store mating start times
        self.sex_ids = {}  # Dictionary to store sex IDs for each ROI
        self.female_fly_positions = {}  # Dictionary to store positions for each ROI
        self.last_known_positions = {}  # Dictionary to store last known positions for flies in each ROI
        self.female_trail_data = {}  # Dictionary to store trail data for female flies
        self.previous_positions = {}  # Dictionary to store last known positions for flies in each ROI
        self.consecutive_size_diff_count = {}

    def export_combined_mating_times(self):
        combined_mating_times = {}
        for roi_id, mating_time in self.mating_start_times.items():
            # Check if this mating time is within 1 second of another mating time
            is_combined = False
            for combined_id, combined_time in combined_mating_times.items():
                if abs(mating_time - combined_time) <= 1:
                    combined_mating_times[combined_id] = (combined_time + mating_time) / 2
                    is_combined = True
                    break
            if not is_combined:
                combined_mating_times[roi_id] = mating_time

        # Create a DataFrame from the combined mating times
        combined_mating_df = pd.DataFrame(list(combined_mating_times.items()), columns=['ROI', 'Start Time'])
        combined_mating_df['Mating Duration'] = [self.mating_durations.get(roi_id, 0) for roi_id in
                                                 combined_mating_df['ROI']]
        return combined_mating_df

    def run(self):
        self.is_running = True

        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        current_frame = 0
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame, self.initial_contours, current_frame)  # or any other value for frame_count
            self.frame_info.emit(current_frame, current_frame / self.fps)
            current_frame += 1

            # Process the frame here
            processed_frame, masks = self.process_frame(frame, self.initial_contours, frame_count)
            self.detect_flies(processed_frame, masks, frame_count)
            self.frame_processed.emit(processed_frame, self.mating_durations)

            frame_count += 1

        cap.release()
        self.finished.emit()

    def stop(self):
        self.is_running = False

    def process_frame(self, frame, initial_contours, frame_count):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold the image to obtain a binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours of white regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to a list and sort based on x-coordinate of their bounding rectangles
        contours_list = list(contours)
        contours_list.sort(key=lambda ctr: cv2.boundingRect(ctr)[0])

        # Emit the binary image
        self.binary_image_processed.emit(thresh)

        # Store initial contours if frame_count <= 100
        if frame_count <= 100:
            initial_contours.clear()
            self.roi_ids.clear()  # Clear ROI IDs
            for i, contour in enumerate(contours_list):
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold to filter out noise
                    initial_contours.append({"contour": contour, "edge_duration": 0})
                    contour_id = self.generate_contour_id(contour)
                    self.roi_ids[contour_id] = i + 1  # Assign ID to ROI
        else:
            # Check for contours near the edges
            for contour_data in initial_contours:
                contour = contour_data["contour"]
                (x, y, w, h) = cv2.boundingRect(contour)
                if x <= 5 or y <= 5 or (x + w) >= frame.shape[1] - 5 or (y + h) >= frame.shape[0] - 5:
                    contour_data["edge_duration"] += 1
                else:
                    contour_data["edge_duration"] = 0

        # Create masks based on the initial contours
        masks = []
        for contour_data in initial_contours:
            mask = np.zeros_like(gray)
            ellipse = cv2.fitEllipse(contour_data["contour"])
            cv2.ellipse(mask, ellipse, 255, -1)
            masks.append(mask)

        # Draw green circles and highlight areas where circles touch the black background
        processed_frame = frame.copy()
        for contour_data in initial_contours:
            contour = contour_data["contour"]
            edge_duration = contour_data["edge_duration"]
            (x, y, w, h) = cv2.boundingRect(contour)
            # Exclude contours near the edges of the frame if edge duration is less than a threshold
            if (x > 5 and y > 5 and (x + w) < frame.shape[1] - 5 and (y + h) < frame.shape[
                0] - 5) or edge_duration >= 90:
                cv2.circle(processed_frame, (int(x + w / 2), int(y + h / 2)), int((w + h) / 4), (0, 255, 0), 2)
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(processed_frame, ellipse, (0, 255, 0), 2)

        return processed_frame, masks

    def detect_flies(self, frame, masks, frame_count):
        # Initialize blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        # Create blob detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Define radius and thickness for drawing circles
        radius = 6  # Increase the radius for larger dots
        thickness = -1  # Set the thickness to a negative value for a hollow circle

        grace_frames_threshold = int(self.fps)  # Number of frames equivalent to 1 second

        # Create a copy of the frame for drawing the pink lines
        overlay_frame = frame.copy()

        # Iterate through each mask and detect flies
        for i, mask in enumerate(masks):
            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert the masked frame to grayscale
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            # Detect blobs (flies)
            keypoints = detector.detect(gray)

            # Initialize previous positions if not available
            if i not in self.previous_positions:
                self.previous_positions[i] = {}

            if len(keypoints) == 2:
                # If this is the first time we've seen this ROI, initialize based on size
                if i not in self.last_known_positions:
                    if keypoints[0].size > keypoints[1].size:
                        self.sex_ids[i] = ('F', 'M')
                        self.last_known_positions[i] = {'F': keypoints[0].pt, 'M': keypoints[1].pt}
                    else:
                        self.sex_ids[i] = ('M', 'F')
                        self.last_known_positions[i] = {'F': keypoints[1].pt, 'M': keypoints[0].pt}
                else:
                    # Use last known positions to identify flies
                    last_f_pos = self.last_known_positions[i]['F']
                    last_m_pos = self.last_known_positions[i]['M']

                    dist_to_last_f_0 = np.linalg.norm(np.array(last_f_pos) - np.array(keypoints[0].pt))
                    dist_to_last_f_1 = np.linalg.norm(np.array(last_f_pos) - np.array(keypoints[1].pt))

                    if dist_to_last_f_0 < dist_to_last_f_1:
                        self.last_known_positions[i] = {'F': keypoints[0].pt, 'M': keypoints[1].pt}
                        self.sex_ids[i] = ('F', 'M')
                    else:
                        self.last_known_positions[i] = {'F': keypoints[1].pt, 'M': keypoints[0].pt}
                        self.sex_ids[i] = ('M', 'F')

                # Draw gender IDs on the frame
                cv2.putText(frame, f"{self.sex_ids.get(i, ('-', '-'))[0]} {1}",
                            (int(keypoints[0].pt[0]), int(keypoints[0].pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"{self.sex_ids.get(i, ('-', '-'))[1]} {2}",
                            (int(keypoints[1].pt[0]), int(keypoints[1].pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Identify the Female Fly and Store Position Data
                if i in self.sex_ids:
                    female_idx = self.sex_ids[i].index('F')  # Find the index of female fly
                    x, y = int(keypoints[female_idx].pt[0]), int(keypoints[female_idx].pt[1])

                    # Store the position data
                    if i not in self.female_fly_positions:
                        self.female_fly_positions[i] = []
                    self.female_fly_positions[i].append((frame_count, x, y))

                    # Store the position data for drawing the pink lines
                    if i not in self.female_trail_data:
                        self.female_trail_data[i] = []
                    self.female_trail_data[i].append({'frame': frame_count, 'x': x, 'y': y})

                    # Draw the trail for female flies on overlay_frame
                    for j in range(1, len(self.female_trail_data[i])):
                        x1, y1 = self.female_trail_data[i][j - 1]['x'], self.female_trail_data[i][j - 1]['y']
                        x2, y2 = self.female_trail_data[i][j]['x'], self.female_trail_data[i][j]['y']
                        cv2.line(overlay_frame, (x1, y1), (x2, y2), (203, 192, 255), 2)  # Pink line

                    # Remove trail points that are older than 100 seconds
                    self.female_trail_data[i] = [point for point in self.female_trail_data[i] if
                                                 frame_count - point['frame'] <= self.fps * 10]

            # Draw dots on the frame for each detected fly (centroid), color depends on mating status
            if len(keypoints) == 1:  # A mating event is occurring
                x = int(keypoints[0].pt[0])
                y = int(keypoints[0].pt[1])

                # Start timing the mating event if not already started
                if i not in self.mating_start_frames:
                    self.mating_start_frames[i] = frame_count

                # Reset grace frames counter for this ROI
                self.mating_grace_frames[i] = 0

                # Calculate the duration of the mating event in frames and convert to seconds
                mating_duration = (frame_count - self.mating_start_frames[i]) / self.fps
                self.mating_durations.setdefault(i, []).append(mating_duration)  # Store duration in list per ROI

                # Change dot color based on mating duration
                if mating_duration < 360:  # Less than 1 minute
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), thickness)  # Yellow dot
                else:  # Over 1 minute
                    cv2.circle(frame, (x, y), radius, (255, 0, 0), thickness)  # Blue dot

                    # If mating duration exceeds 60 seconds and this ROI doesn't have a verified mating start time yet
                    if i not in self.mating_start_times:
                        mating_time = frame_count / self.fps
                        self.mating_start_times[i] = mating_time
                        # Emit the verified mating start times
                        self.verified_mating_start_times.emit(self.mating_start_times)

                self.fly_info.emit(i, "Mating")
            else:  # Mating event has potentially ended
                # Increase grace frames counter for this ROI
                self.mating_grace_frames[i] = self.mating_grace_frames.get(i, 0) + 1

                # If grace frames counter exceeds threshold, consider the mating event to have ended
                if self.mating_grace_frames[i] > grace_frames_threshold:
                    if i in self.mating_start_frames:
                        del self.mating_start_frames[i]
                        del self.mating_grace_frames[i]

                fly_data = []
                pink_color = (203, 192, 255)  # BGR format for pink
                if i not in self.last_known_positions:
                    self.last_known_positions[i] = {}

                for idx, keypoint in enumerate(keypoints):
                    fly_id = f'Fly_{idx + 1}'
                    x = int(keypoint.pt[0])
                    y = int(keypoint.pt[1])
                    self.last_known_positions[i][fly_id] = {'x': x, 'y': y}
                    size = keypoint.size

                    # Draw a circle around the detected keypoint
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red dot

                    # Create a mask for this keypoint
                    keypoint_mask = np.zeros_like(frame[:, :, 0])
                    cv2.circle(keypoint_mask, (x, y), int(size / 2), 255, -1)

                    # Apply pink color to the fly region
                    # frame[keypoint_mask == 255] = pink_color

                    # Store the fly data for this ROI
                    fly_data.append({'x': x, 'y': y, 'size': size})

                self.fly_info.emit(i, fly_data)

        # Combine the original frame and the overlay
        alpha = 0.5  # Define the alpha for blending, adjust as needed
        cv2.addWeighted(overlay_frame, alpha, frame, 1 - alpha, 0, frame)
        self.frame_processed.emit(frame, self.mating_durations)  # Emit the frame and the mating durations

    def generate_contour_id(self, contour):
        return cv2.contourArea(contour)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fly Behavior Analysis")
        self.setGeometry(200, 200, 1200, 700)

        self.video_path = None
        self.initial_contours = []

        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 10, 780, 440)

        self.mating_duration_label = QLabel(self)
        self.mating_duration_label.setGeometry(10, 360, 700, 150)
        self.binary_image_label = QLabel(self)
        self.binary_image_label.setGeometry(800, -30, 380, 270)  # Set the geometry as per your layout

        self.frame_label = QLabel('Frame: 0', self)
        self.frame_label.move(200, 410)
        self.time_label = QLabel('Time (s): 0', self)
        self.time_label.move(200, 430)
        self.verified_mating_times_label = QLabel(self)
        self.verified_mating_times_label.setGeometry(330, 350, 780, 120)

        self.fly_info_label = QLabel(self)
        self.fly_info_label.setGeometry(800, 130, 780, 250)
        self.all_fly_info = {}  # Dictionary to store fly information for all ROIs

        self.fps_input = QLineEdit(self)
        self.fps_input.setGeometry(10, 630, 780, 30)
        self.fps_input.setPlaceholderText("Enter Video FPS")

        self.select_button = QPushButton("Select Video", self)
        self.select_button.setGeometry(10, 510, 780, 30)
        self.select_button.clicked.connect(self.select_video)

        self.start_button = QPushButton("Start Processing", self)
        self.start_button.setGeometry(10, 550, 780, 30)
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)  # The button is initially disabled

        self.stop_button = QPushButton("Stop Processing", self)
        self.processing_status_label = QLabel(self)
        self.processing_status_label.setGeometry(10, 670, 780, 30)
        self.stop_button.setGeometry(10, 590, 780, 30)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)  # The button is initially disabled

        self.video_thread = None
        self.add_export_button()  # Add this line to create the "Export DataFrame" button

    def add_export_button(self):
        self.export_button = QPushButton("Export DataFrame", self)
        self.export_button.setGeometry(10, 480, 780, 30)
        self.export_button.clicked.connect(self.export_dataframe)
        self.export_button.setEnabled(False)  # The button is initially disabled

    def enable_export_button(self):
        self.export_button.setEnabled(True)

    def export_dataframe(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export DataFrame to CSV", "",
                                                   "CSV Files (*.csv);;All Files (*)")
        if file_path:
            mating_start_times = self.video_thread.mating_start_times

            # Adjust mating start times by subtracting 360 seconds
            adjusted_mating_start_times = {roi_id: max(0, time - 360) for roi_id, time in mating_start_times.items()}

            mating_times_df = pd.DataFrame(adjusted_mating_start_times.items(), columns=['ROI', 'Start Time'])

            # Calculate longest mating duration for each ROI from the stored lists
            longest_mating_durations = {}
            for roi_id, durations in self.video_thread.mating_durations.items():
                longest_mating_durations[roi_id] = max(durations, default=0)  # Use max() with default for empty lists

            mating_times_df['Longest Duration'] = [longest_mating_durations.get(roi_id, 0) for roi_id in
                                                   mating_times_df['ROI']]

            mating_times_df.to_csv(file_path, index=False)
            self.processing_status_label.setText('DataFrame exported successfully.')

    def update_verified_mating_times(self, mating_times_dict):
        # Subtract 360 seconds from the verified mating start times
        adjusted_mating_times_dict = {roi_id: max(0, time - 360) for roi_id, time in mating_times_dict.items()}

        # Update verified mating times in the DataFrame
        self.mating_start_times_df = pd.DataFrame(list(adjusted_mating_times_dict.items()), columns=['ROI', 'Start Time'])
        self.mating_start_times_df['Mating Duration'] = [self.video_thread.mating_durations.get(roi_id, 0) for
                                                         roi_id in self.mating_start_times_df['ROI']]

        # Call the method to enable the "Export DataFrame" button
        self.enable_export_button()

        # Update verified mating times in the GUI
        mating_time_text = "\n".join(
            [f"ROI {roi_id}: {time:.2f} seconds" for roi_id, time in adjusted_mating_times_dict.items()])
        self.verified_mating_times_label.setText(mating_time_text)

    def update_fly_info(self, roi_id, fly_data):
        if isinstance(fly_data, str) and fly_data == "Mating":
            self.all_fly_info[roi_id] = "Mating"
        else:
            self.all_fly_info[roi_id] = fly_data

        info_text = ""
        for roi_id, flies in self.all_fly_info.items():
            if flies == "Mating":
                info_text += f"ROI {roi_id} - Mating\n"
            else:
                genders = self.video_thread.sex_ids.get(roi_id, ('-', '-'))  # Get genders for this ROI
                for idx, (fly, gender) in enumerate(zip(flies, genders)):
                    info_text += f"ROI {roi_id} - Fly {idx + 1} (Gender: {gender}): Size = {fly['size']:.2f}, Centroid = ({fly['x']}, {fly['y']})\n"

        self.fly_info_label.setText(info_text)

    def select_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video")
        self.initial_contours.clear()
        if self.video_path:
            # Automatically find and set the FPS
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()  # Don't forget to release the VideoCapture object
            self.fps_input.setText(str(fps))  # Set the text of the fps_input QLineEdit
            self.start_button.setEnabled(True)  # Enable the start button

    def update_binary_image(self, binary_image):
        # Convert the single channel binary image to a 3-channel image
        # This makes it compatible with QImage.Format_RGB888
        three_channel_binary = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        height, width, channel = three_channel_binary.shape
        bytes_per_line = 3 * width
        q_img = QImage(three_channel_binary.data, width, height, bytes_per_line,
                       QImage.Format.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.binary_image_label.width(), self.binary_image_label.height(),
                               Qt.AspectRatioMode.KeepAspectRatio)
        self.binary_image_label.setPixmap(pixmap)

    def start_processing(self):
        self.all_fly_info.clear()  # Clear the accumulated fly information
        if self.video_path and self.fps_input.text():
            self.start_button.setEnabled(False)
            self.select_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            fps = float(self.fps_input.text())
            self.video_thread = VideoProcessingThread(self.video_path, self.initial_contours, fps)
            self.video_thread.verified_mating_start_times.connect(self.update_verified_mating_times)
            self.video_thread.binary_image_processed.connect(self.update_binary_image)
            self.video_thread.fly_info.connect(self.update_fly_info)
            self.video_thread.frame_info.connect(self.update_frame_info)
            self.video_thread.frame_processed.connect(self.update_video_frame)
            self.video_thread.finished.connect(self.processing_finished)
            self.video_thread.start()

    def stop_processing(self):
        self.all_fly_info.clear()  # Clear the accumulated fly information
        if self.video_thread.isRunning():
            self.video_thread.stop()

    def processing_finished(self):
        self.processing_status_label.setText('Video processing finished.')
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_video_frame(self, frame, mating_durations):
        # Update video label
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

        # Update mating duration label with the newest duration for each ROI
        mating_duration_text = ""
        for roi_id, durations in mating_durations.items():
            latest_duration = durations[-1] if durations else 0
            mating_duration_text += f"ROI {roi_id}: {latest_duration:.2f} seconds\n"

        self.mating_duration_label.setText(mating_duration_text)

    def update_frame_info(self, frame, time):
        self.frame_label.setText(f'Frame: {frame}')
        self.time_label.setText(f'Time (s): {time:.2f}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())