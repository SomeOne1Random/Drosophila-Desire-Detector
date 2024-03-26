import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QLineEdit, QListWidget,
                             QHBoxLayout, QMessageBox, QGroupBox,  QScrollArea, QVBoxLayout, QWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import pandas as pd
from scipy import stats
import os


class VideoProcessingThread(QThread):
    finished = pyqtSignal()
    frame_processed = pyqtSignal(str, np.ndarray, dict)
    frame_info = pyqtSignal(int, float)
    verified_mating_start_times = pyqtSignal(str, dict)
    void_roi_signal = pyqtSignal(str, int)  # Signal to emit with video_path and void ROI ID
    mating_analysis_complete = pyqtSignal(str)  # Signal to indicate completion of mating analysis
    center_mating_duration_signal = pyqtSignal(int, float)  # ROI ID and duration in seconds
    center_gender_duration_signal = pyqtSignal(int, float, float)  # ROI ID, male duration, female duration

    def __init__(self, video_path, initial_contours, fps, skip_frames=0, perf_frame_skips=1):
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
        self.latest_frames = {}  # Stores the latest frame for each video
        self.latest_mating_durations = {}  # Stores the latest mating durations for each video
        self.flies_count_signal = pyqtSignal(str, int,
                                             int)  # Signal to be emitted with video_path, ROI ID, and flies count
        self.flies_count_per_ROI = {}  # Tracks the count of flies per ROI
        self.void_rois = {}  # Dictionary to store void ROIs
        self.skip_frames = skip_frames  # Number of frames to skip from the beginning
        self.previous_flies_count_per_ROI = {}  # Tracks previous frame's fly count per ROI
        self.mating_event_detected = {}  # Tracks if a mating event is detected in an ROI
        self.previous_fly_positions_per_ROI = {}  # Tracks previous frame's fly positions per ROI when there are two flies
        self.mating_status_per_ROI = {}  # New dictionary to store mating status for each ROI
        self.mating_event_ongoing = {}  # Tracks ongoing mating events for each ROI
        self.perf_frame_skips = perf_frame_skips
        self.roi_centers = {}
        self.center_mating_frames_count = {}  # Tracks count of center-mating frames for each ROI
        self.center_mating_duration = {}  # Dictionary to store the longest center mating duration for each ROI
        self.center_mating_start_frame = {}
        self.center_mating_event_end_threshold = 3
        self.fly_size_history = {}
        self.fly_position_history = {}
        self.fly_trail_history = {}
        self.center_gender_duration = {}  # Initialize the dictionary to store center duration for each gender


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
        self.flies_count_per_ROI.clear()  # Reset flies count per ROI when a new video starts

        cap = cv2.VideoCapture(self.video_path)

        # Skip the specified number of frames
        for _ in range(self.skip_frames):
            ret, _ = cap.read()
            if not ret:
                break

        frame_count = 0
        current_frame = 0
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % self.perf_frame_skips == 0:

                self.process_frame(frame, self.initial_contours, current_frame)  # or any other value for frame_count
                self.frame_info.emit(current_frame, current_frame / self.fps)

                # Process the frame here
                processed_frame, masks = self.process_frame(frame, self.initial_contours, frame_count)
                self.detect_flies(processed_frame, masks, frame_count)

                # Emit the video path along with the processed frame and the mating durations
                self.frame_processed.emit(self.video_path, processed_frame, self.mating_durations)

                for roi_id, is_mating in self.mating_event_ongoing.items():
                    self.mating_status_per_ROI[roi_id] = is_mating

            current_frame += 1
            frame_count += 1

        self.mating_analysis_complete.emit(self.video_path)  # Emit signal indicating completion

        cap.release()
        self.finished.emit()

    def stop(self):
        self.is_running = False

    def process_frame(self, frame, initial_contours, frame_count):
        # Define padding size (you can adjust these values as needed)
        top_padding, bottom_padding, left_padding, right_padding = 50, 50, 50, 50  # Example padding sizes

        # Add black padding to the frame
        frame_with_padding = cv2.copyMakeBorder(frame, top_padding, bottom_padding, left_padding, right_padding,
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame_with_padding, cv2.COLOR_BGR2GRAY)

        # Threshold the image to obtain a binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours of white regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Define a custom sorting function
        def custom_sort(contour):
            x, y, w, h = cv2.boundingRect(contour)
            y_tolerance = 200  # Adjust this tolerance as needed
            return (y // y_tolerance) * 1000 + x  # Sort primarily by y (with tolerance), and then by x

        # Convert contours to a list and sort based on x-coordinate of their bounding rectangles
        contours_list = list(contours)

        # Sort the contours using the custom sorting function
        contours_list.sort(key=custom_sort)

        # Store initial contours if frame_count
        if frame_count <= 500:
            initial_contours.clear()
            self.roi_ids.clear()  # Clear ROI IDs
            for i, contour in enumerate(contours_list):
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold to filter out noise
                    initial_contours.append({"contour": contour, "edge_duration": 0})
                    contour_id = self.generate_contour_id(contour)
                    self.roi_ids[contour_id] = i + 1  # Assign ID to ROI
        else:
            # Check for contours near the edges
            for contour_data in initial_contours:
                contour = contour_data["contour"]
                (x, y, w, h) = cv2.boundingRect(contour)
                if x <= 5 or y <= 5 or (x + w) >= frame_with_padding.shape[1] - 5 or (y + h) >= \
                        frame_with_padding.shape[0] - 5:
                    contour_data["edge_duration"] += 1
                else:
                    contour_data["edge_duration"] = 0

        # Calculate and round radii to find the mode radius
        radii = []
        for contour_data in initial_contours:
            (x, y, w, h) = cv2.boundingRect(contour_data["contour"])
            radii.append(int(round((w + h) / 4)))

        mode_radius = int(stats.mode(radii)[0]) if radii else 0  # Default to 0 if radii list is empty

        # Create masks and draw green circles using mode radius
        masks = []
        processed_frame = frame_with_padding.copy()
        for contour_data in initial_contours:
            (x, y, w, h) = cv2.boundingRect(contour_data["contour"])
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # Create mask with circle
            mask = np.zeros(processed_frame.shape[:2], dtype="uint8")
            cv2.circle(mask, (center_x, center_y), mode_radius, (255,), -1)
            masks.append(mask)

            # Draw circle on processed frame
            if (x > 5 and y > 5 and (x + w) < frame_with_padding.shape[1] - 5 and (y + h) < frame_with_padding.shape[
                0] - 5) or contour_data["edge_duration"] >= 90:
                cv2.circle(processed_frame, (center_x, center_y), mode_radius, (0, 255, 0), 2)

        # Draw ROI numbers
        for i, contour_data in enumerate(initial_contours):
            (x, y, w, h) = cv2.boundingRect(contour_data["contour"])
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # Inside process_frame, in the loop where circles are drawn
            self.roi_centers[i] = (center_x, center_y)  # Store the center for each ROI

            # Determine position for ROI number
            text_position = (center_x, center_y - 55)
            cv2.putText(processed_frame, str(i), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 2,
                        cv2.LINE_AA)

        return processed_frame, masks

    def detect_flies(self, frame_with_padding, masks, frame_count):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 1
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        # Create blob detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Define radius and thickness for drawing circles
        radius = 6  # Increase the radius for larger dots
        thickness = -1  # Set the thickness to a negative value for a hollow circle

        grace_frames_threshold = int(self.fps * 3 / self.perf_frame_skips)  # Assuming 1 second of real-time

        center_threshold = 32  # Define a threshold for how close to the center is considered 'in the center'

        # Iterate through each mask and detect flies
        for i, mask in enumerate(masks):
            # If an ROI has been marked as void, continue to the next ROI
            if self.void_rois.get(i, False):
                continue

            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame_with_padding, frame_with_padding, mask=mask)

            # Convert the masked frame to grayscale (if not already done)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((5, 5), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # Continue with the existing blob detection
            keypoints = detector.detect(gray)

            # Initialize or update the trail history for each ROI
            if i not in self.fly_trail_history:
                self.fly_trail_history[i] = []

            # Get the center of the current ROI
            roi_center = self.roi_centers.get(i, (0, 0))

            # Draw the threshold boundary around the center of the ROI
            cv2.circle(frame_with_padding, roi_center, center_threshold, (255, 0, 255), 2)  # Drawing a magenta circle

            # Detect flies count for the first 500 frames
            if frame_count < 500:
                flies_count = len(keypoints)
                current_positions = [keypoint.pt for keypoint in keypoints]

                # Check for mating event when there's a transition from two to one fly
                if flies_count == 1 and i in self.previous_fly_positions_per_ROI:
                    prev_positions = self.previous_fly_positions_per_ROI[i]
                    # Ensure prev_positions has exactly two elements
                    if len(prev_positions) == 2:
                        distance_between_flies = np.linalg.norm(
                            np.array(prev_positions[0]) - np.array(prev_positions[1]))
                        if distance_between_flies > 30:
                            self.mating_event_detected[i] = True
                    # Clear the stored positions after checking
                    del self.previous_fly_positions_per_ROI[i]

                # Only store positions when there are exactly two flies
                elif flies_count == 2:
                    self.previous_fly_positions_per_ROI[i] = current_positions

                if i not in self.flies_count_per_ROI:
                    self.flies_count_per_ROI[i] = []
                self.flies_count_per_ROI[i].append(flies_count)

                self.previous_fly_positions_per_ROI[i] = current_positions

                # Check the condition after 200 frames
                if len(self.flies_count_per_ROI[i]) == 200:
                    more_than_two_count = sum(count > 2 for count in self.flies_count_per_ROI[i])
                    less_than_two_count = sum(count < 2 for count in self.flies_count_per_ROI[i])

                    # Calculate 75% of 200 frames
                    threshold = 200 * 0.75

                    # Adjust the logic to not mark the ROI as void if a mating event is detected
                    if more_than_two_count > threshold or (
                            less_than_two_count > threshold and not self.mating_event_detected.get(i, False)):
                        self.void_rois[i] = True
                        self.void_roi_signal.emit(self.video_path, i)  # Emit the signal

                # Mating event detection and handling
            if len(keypoints) == 1:  # A mating event is occurring
                if self.mating_durations.get(i, []) and max(self.mating_durations[i]) >= 360 and \
                        not self.mating_event_ongoing[i] and self.mating_grace_frames.get(i,
                                                                                          0) > grace_frames_threshold:
                    continue

                self.mating_event_ongoing[i] = True

                x, y = int(keypoints[0].pt[0]), int(keypoints[0].pt[1])
                self.fly_trail_history[i].append((x, y))

                # Draw the trail if there are enough points
                if len(self.fly_trail_history[i]) > 1:
                    for j in range(len(self.fly_trail_history[i]) - 1):
                        # Draw lines connecting consecutive points
                        cv2.line(frame_with_padding, self.fly_trail_history[i][j], self.fly_trail_history[i][j + 1],
                                 (0, 255, 0), 2)

                # Start timing the mating event if not already started
                if i not in self.mating_start_frames:
                    self.mating_start_frames[i] = frame_count

                # Reset grace frames counter for this ROI
                self.mating_grace_frames[i] = 0

                # Calculate the duration of the mating event in frames and convert to seconds
                mating_duration = (frame_count - self.mating_start_frames[i]) / self.fps
                self.mating_durations.setdefault(i, []).append(mating_duration)  # Store duration in list per ROI

                # If mating duration exceeds 60 seconds and this ROI doesn't have a verified mating start time yet
                if mating_duration >= 360 and i not in self.mating_start_times:
                    self.mating_durations[i] = [max(self.mating_durations[
                                                        i])]  # Keep the longest duration only
                    mating_time = frame_count / self.fps
                    self.mating_start_times[i] = mating_time
                    # Emit the verified mating start times
                    self.verified_mating_start_times.emit(self.video_path, self.mating_start_times)

                x, y = int(keypoints[0].pt[0]), int(keypoints[0].pt[1])
                distance_to_center = np.sqrt((x - roi_center[0]) ** 2 + (y - roi_center[1]) ** 2)
                in_center = distance_to_center <= center_threshold

                if in_center:
                    if i not in self.center_mating_start_frame:
                        self.center_mating_start_frame[i] = frame_count
                    else:
                        # Calculate the duration of the center mating event
                        duration = frame_count - self.center_mating_start_frame[i]
                        # Convert duration to seconds
                        duration_in_seconds = duration / self.fps

                        # Ensure the key exists
                        if i not in self.center_mating_duration:
                            self.center_mating_duration[i] = []

                        # Check if the mating event is still ongoing
                        if self.mating_event_ongoing[i]:
                            # Now append the duration
                            self.center_mating_duration[i].append(duration_in_seconds)

                            # Sum up all durations for the current ROI to get the total duration, if the key exists
                            if i in self.center_mating_duration:
                                total_duration = sum(self.center_mating_duration[i])
                                self.center_mating_duration_signal.emit(i, total_duration)

                        # Reset the start frame for the next center mating event
                        self.center_mating_start_frame[i] = frame_count
                else:
                    # Check if the center mating event has exceeded the threshold
                    if i in self.center_mating_start_frame:
                        duration_since_last_event = frame_count - self.center_mating_start_frame[i]
                        if duration_since_last_event > self.center_mating_event_end_threshold:
                            # Reset the start frame for the next event, instead of deleting it
                            self.center_mating_start_frame[i] = frame_count
                            self.fly_trail_history[i] = []
                            self.mating_event_ongoing[i] = False

                # Track center mating duration
                x, y = int(keypoints[0].pt[0]), int(keypoints[0].pt[1])
                distance_to_center = np.sqrt((x - roi_center[0]) ** 2 + (y - roi_center[1]) ** 2)
                in_center = distance_to_center <= center_threshold

                if in_center:
                    if i not in self.center_mating_start_frame:
                        self.center_mating_start_frame[i] = frame_count
                    else:
                        # Calculate the duration of the center mating event
                        center_duration = (frame_count - self.center_mating_start_frame[i]) / self.fps

                        if mating_duration >= 360:
                            self.center_mating_duration.setdefault(i, []).append(center_duration)

                        # Update the start frame for the next center mating event
                        self.center_mating_start_frame[i] = frame_count
                else:
                    if i in self.center_mating_start_frame:
                        # Check if the center mating event has exceeded the threshold
                        duration_since_last_event = (frame_count - self.center_mating_start_frame[i]) / self.fps
                        if duration_since_last_event > self.center_mating_event_end_threshold:
                            # Reset the start frame for the next event
                            self.center_mating_start_frame[i] = frame_count
                            self.fly_trail_history[i] = []
                            self.mating_event_ongoing[i] = False

            else:  # Mating event has potentially ended
                self.mating_event_ongoing[i] = False
                self.mating_grace_frames[i] = self.mating_grace_frames.get(i, 0) + 1

                # If grace frames counter exceeds threshold, consider the mating event to have ended
                if self.mating_grace_frames[i] > grace_frames_threshold:
                    if i in self.mating_start_frames:
                        del self.mating_start_frames[i]
                        del self.mating_grace_frames[i]

                # Initialize or update tracking information
            if i not in self.fly_size_history:
                self.fly_size_history[i] = {'male': [], 'female': []}  # Stores size history for male and female
                self.fly_position_history[i] = {'female': []}  # Stores position history for the female fly

            if len(keypoints) == 2:
                # Sort keypoints by size (area)
                sorted_keypoints = sorted(keypoints, key=lambda k: k.size, reverse=True)
                female_fly, male_fly = sorted_keypoints

                # Update size history
                self.fly_size_history[i]['female'].append(female_fly.size)
                self.fly_size_history[i]['male'].append(male_fly.size)

                # Maintain size history only for the last N frames
                size_history_limit = 20
                for gender in ['male', 'female']:
                    if len(self.fly_size_history[i][gender]) > size_history_limit:
                        self.fly_size_history[i][gender].pop(0)

                # Determine gender based on average size over history
                average_female_size = np.mean(self.fly_size_history[i]['female'])
                average_male_size = np.mean(self.fly_size_history[i]['male'])

                if average_female_size > average_male_size:
                    # Update female fly position history
                    self.fly_position_history[i]['female'].append((int(female_fly.pt[0]), int(female_fly.pt[1])))
                else:
                    # If the male becomes larger, swap the labels
                    self.fly_position_history[i]['female'].append((int(male_fly.pt[0]), int(male_fly.pt[1])))

                # Maintain position history only for the last 100 frames
                if len(self.fly_position_history[i]['female']) > 10:
                    self.fly_position_history[i]['female'].pop(0)

                # Draw lines for the female fly
                for p1, p2 in zip(self.fly_position_history[i]['female'], self.fly_position_history[i]['female'][1:]):
                    cv2.line(frame_with_padding, p1, p2, (255, 0, 0), 2)  # Blue line for the female fly

                # Draw keypoints with updated gender labels
                cv2.circle(frame_with_padding, (int(female_fly.pt[0]), int(female_fly.pt[1])), radius, (0, 0, 255),
                           thickness)  # Red for female
                cv2.circle(frame_with_padding, (int(male_fly.pt[0]), int(male_fly.pt[1])), radius, (255, 255, 0),
                           thickness)  # Yellow for male

                for fly, gender in zip(sorted_keypoints, ['female', 'male']):
                    x, y = int(fly.pt[0]), int(fly.pt[1])
                    distance_to_center = np.sqrt((x - roi_center[0]) ** 2 + (y - roi_center[1]) ** 2)
                    in_center = distance_to_center <= center_threshold

                    # Initialize the nested dictionary if necessary
                    if i not in self.center_gender_duration:
                        self.center_gender_duration[i] = {'male': 0, 'female': 0}

                    if in_center:
                        # Increment the center duration for the respective gender
                        self.center_gender_duration[i][gender] += 1 / self.fps  # Convert frame count to seconds

                        # Emit the center gender duration signal
                        male_duration = self.center_gender_duration[i]['male']
                        female_duration = self.center_gender_duration[i]['female']
                        self.center_gender_duration_signal.emit(i, male_duration, female_duration)


            # Draw dots on the frame for each detected fly (centroid)
            for keypoint in keypoints:
                x = int(keypoint.pt[0])
                y = int(keypoint.pt[1])

                # Calculate the distance from the fly to the center of the ROI
                distance_to_center = np.sqrt((x - roi_center[0]) ** 2 + (y - roi_center[1]) ** 2)

                # Determine the color based on mating status, grace frame status, and proximity to center
                in_center = distance_to_center <= center_threshold
                mating_ongoing = self.mating_event_ongoing.get(i, False)
                within_grace_period = self.mating_grace_frames.get(i, 0) <= grace_frames_threshold

                if mating_ongoing or within_grace_period:
                    mating_duration = self.mating_durations.get(i, [0])[-1]  # Get the latest duration

                    if mating_duration < 360:
                        color = (0, 255, 255)  # Yellow dot
                    else:
                        color = (255, 0, 0)  # Blue dot
                else:
                    color = (0, 0, 255)  # Red dot for flies not in mating or grace period

                # Change color if the fly is in the center
                if in_center:
                    color = (0, 255, 0)  # Green color for flies in the center

                cv2.circle(frame_with_padding, (x, y), radius, color, thickness)

        self.frame_processed.emit(self.video_path, frame_with_padding, self.mating_durations)

    def generate_contour_id(self, contour):
        return cv2.contourArea(contour)

    def void_roi(self, roi_id):
        self.void_rois[roi_id] = True
        self.void_roi_signal.emit(self.video_path, roi_id)  # Emit the signal to indicate a void ROI


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window attributes
        self.setWindowTitle("Fly Behavior Analysis")
        self.setGeometry(200, 200, 1200, 1400)  # Adjust size as needed

        # Paths and initial setups
        self.video_path = None
        self.initial_contours = []
        self.video_paths = []  # List to store multiple video paths
        self.video_threads = {}  # Dictionary to store threads for each video path
        self.current_video_index = 0  # Index to keep track of the currently displayed video
        self.latest_frames = {}  # Stores the latest frame for each video
        self.latest_mating_durations = {}  # Stores the latest mating durations for each video
        self.mating_start_times_dfs = {}  # Dictionary to store mating start times for each video
        self.center_gender_duration_labels = {}

        # Organize UI elements
        self.init_ui()
        self.auto_export_directory = "path_to_export_directory"  # Set a default directory for auto-export


    def init_ui(self):
        # Video Display & Info Section
        video_display_group = QGroupBox("Video Display", self)
        video_display_group.setGeometry(10, 10, 870, 500)

        vbox = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setFixedSize(860, 440)
        vbox.addWidget(self.video_label)

        hbox = QHBoxLayout()
        self.frame_label = QLabel('Frame: 0')
        hbox.addWidget(self.frame_label)
        self.time_label = QLabel('Time (s): 0')
        hbox.addWidget(self.time_label)
        vbox.addLayout(hbox)

        video_display_group.setLayout(vbox)

        # Video Control Section
        video_control_group = QGroupBox("Video Controls", self)
        video_control_group.setGeometry(10, 520, 870, 110)

        vbox = QVBoxLayout()

        self.fps_input = QLineEdit()
        self.fps_input.setPlaceholderText("Enter Video FPS")
        vbox.addWidget(self.fps_input)

        hbox = QHBoxLayout()
        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        hbox.addWidget(self.select_button)

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        hbox.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        hbox.addWidget(self.stop_button)

        vbox.addLayout(hbox)
        video_control_group.setLayout(vbox)

        # Video List Section
        video_list_group = QGroupBox("Video List", self)
        video_list_group.setGeometry(10, 640, 870, 120)

        vbox = QVBoxLayout()

        self.video_list_widget = QListWidget()
        vbox.addWidget(self.video_list_widget)

        video_list_group.setLayout(vbox)

        # Mating Information Display Area
        mating_info_area = QWidget(self)
        mating_info_area.setGeometry(10, 750, 870, 150)

        hbox = QHBoxLayout()

        # Mating Duration Display with Scrollable Area
        mating_duration_group = QGroupBox("Mating Durations", mating_info_area)
        vbox = QVBoxLayout()
        self.mating_duration_label = QLabel("Mating Durations:")
        vbox.addWidget(self.mating_duration_label)

        # Create a scroll area for mating durations
        mating_duration_scroll = QScrollArea()
        mating_duration_scroll.setWidgetResizable(True)
        mating_duration_scroll.setWidget(mating_duration_group)
        mating_duration_group.setLayout(vbox)
        hbox.addWidget(mating_duration_scroll)

        # Verified Mating Times Display with Scrollable Area
        verified_times_group = QGroupBox("Verified Mating Times", mating_info_area)
        vbox = QVBoxLayout()
        self.verified_mating_times_label = QLabel("Verified Mating Times:")
        vbox.addWidget(self.verified_mating_times_label)

        # Create a scroll area for verified mating times
        verified_times_scroll = QScrollArea()
        verified_times_scroll.setWidgetResizable(True)
        verified_times_scroll.setWidget(verified_times_group)
        verified_times_group.setLayout(vbox)
        hbox.addWidget(verified_times_scroll)

        mating_info_area.setLayout(hbox)

        # Navigation Controls
        nav_group = QGroupBox("Navigation", self)
        nav_group.setGeometry(10, 900, 870, 80)

        hbox = QHBoxLayout()

        # Use arrow icons for previous and next buttons
        self.prev_button = QPushButton("← Previous Video")
        self.prev_button.clicked.connect(self.previous_video)
        hbox.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Video →")
        self.next_button.clicked.connect(self.next_video)
        hbox.addWidget(self.next_button)

        nav_group.setLayout(hbox)

        # Export Functionality
        export_group = QGroupBox("Data Export", self)
        export_group.setGeometry(10, 965, 870, 80)

        self.skip_frames_input = QLineEdit(self)
        self.skip_frames_input.setPlaceholderText("Enter number of seconds to skip")
        self.skip_frames_input.setGeometry(1000, 20, 200, 30)  # x, y, width, height

        hbox = QHBoxLayout()

        self.export_button = QPushButton("Export DataFrame")
        self.export_button.clicked.connect(self.export_dataframe)
        self.export_button.setToolTip("Export the mating data as a CSV file.")
        hbox.addWidget(self.export_button)

        self.processing_status_label = QLabel("Status: Awaiting action.")
        hbox.addWidget(self.processing_status_label)

        export_group.setLayout(hbox)


        self.roi_control_group = QGroupBox("Manual ROI Control", self)
        self.roi_control_group.setGeometry(890, 35, 300, 80)  # Adjust the position and size as needed

        roi_control_layout = QHBoxLayout()

        # Modify or replace the existing ROI control group
        self.roi_control_group = QGroupBox("Manual ROI Control", self)
        self.roi_control_group.setGeometry(890, 35, 300, 200)  # Adjust the position and size as needed

        roi_control_layout = QVBoxLayout()  # Changed to QVBoxLayout for better alignment

        # Add a QLineEdit for multiple ROI IDs
        self.multi_roi_input = QLineEdit(self)
        self.multi_roi_input.setPlaceholderText("Enter multiple ROI IDs separated by commas")
        roi_control_layout.addWidget(self.multi_roi_input)

        # Add a button for voiding multiple ROIs
        self.void_multi_roi_button = QPushButton("Void Multiple ROIs", self)
        self.void_multi_roi_button.clicked.connect(self.void_multiple_rois)
        roi_control_layout.addWidget(self.void_multi_roi_button)

        # Add a QListWidget to display ROI voiding status
        self.roi_void_list = QListWidget(self)
        roi_control_layout.addWidget(self.roi_void_list)

        self.roi_control_group.setLayout(roi_control_layout)

        # Add an input for frame skip value
        self.frame_skip_input = QLineEdit(self)
        self.frame_skip_input.setPlaceholderText("Enter Frame Skip Value")

        # Set the geometry of the frame skip input (x, y, width, height)
        self.frame_skip_input.setGeometry(890, 230, 160, 30)  # Adjust these values as needed

        # Initialize the list to store center mating duration labels
        self.center_mating_duration_labels = []

        # Center Mating Duration Display Area
        self.center_mating_duration_group = QGroupBox("Center Mating Duration", self)
        self.center_mating_duration_group.setGeometry(890, 280, 300, 300)

        self.center_mating_duration_layout = QVBoxLayout()
        self.center_mating_duration_group.setLayout(self.center_mating_duration_layout)

        self.scroll_widget_for_center_mating_duration = QWidget()
        self.scroll_layout_for_center_mating_duration = QVBoxLayout(self.scroll_widget_for_center_mating_duration)

        self.scroll_area_for_center_mating_duration = QScrollArea()
        self.scroll_area_for_center_mating_duration.setWidgetResizable(True)
        self.scroll_area_for_center_mating_duration.setWidget(self.scroll_widget_for_center_mating_duration)

        self.center_mating_duration_layout.addWidget(self.scroll_area_for_center_mating_duration)

        # Center Gender Duration Display Area
        self.center_gender_duration_group = QGroupBox("Center Gender Duration", self)
        self.center_gender_duration_group.setGeometry(890, 580, 300, 300)

        self.center_gender_duration_layout = QVBoxLayout()
        self.center_gender_duration_group.setLayout(self.center_gender_duration_layout)

        self.scroll_widget_for_center_gender_duration = QWidget()
        self.scroll_layout_for_center_gating_duration = QVBoxLayout(self.scroll_widget_for_center_gender_duration)

        self.scroll_area_for_center_gender_duration = QScrollArea()
        self.scroll_area_for_center_gender_duration.setWidgetResizable(True)
        self.scroll_area_for_center_gender_duration.setWidget(self.scroll_widget_for_center_gender_duration)

        self.center_gender_duration_layout.addWidget(self.scroll_area_for_center_gender_duration)

    # Handle errors or other information that needs to be shown to the user
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_info(self, title, message):
        QMessageBox.information(self, title, message)

    def void_roi(self):
        try:
            roi_id = int(self.roi_id_input.text())
            current_video_path = self.video_paths[self.current_video_index]
            video_thread = self.video_threads.get(current_video_path)
            if video_thread:
                video_thread.void_roi(roi_id)
                print(f"Manually voided ROI {roi_id} in video {current_video_path}")
            else:
                self.show_error("No video thread found for the current video.")
        except ValueError:
            self.show_error("Invalid ROI ID entered.")

    def void_multiple_rois(self):
        roi_input = self.multi_roi_input.text().strip()  # Get input text and remove leading/trailing whitespace
        roi_ids = []

        # Split the input by commas to get individual entries
        roi_entries = roi_input.split(',')

        for entry in roi_entries:
            entry = entry.strip()  # Remove leading/trailing whitespace from each entry

            # Check if the entry contains a dash to indicate a range
            if '-' in entry:
                start, end = map(int, entry.split('-'))
                roi_ids.extend(range(start, end + 1))
            else:
                try:
                    roi_id = int(entry)
                    roi_ids.append(roi_id)
                except ValueError:
                    self.show_error(f"Invalid ROI ID or range: {entry}")

        current_video_path = self.video_paths[self.current_video_index]
        video_thread = self.video_threads.get(current_video_path)

        if video_thread:
            for roi_id in roi_ids:
                video_thread.void_roi(roi_id)
                self.roi_void_list.addItem(f"ROI {roi_id} voided in video {current_video_path}")
                print(f"Manually voided ROI {roi_id} in video {current_video_path}")
        else:
            self.show_error("No video thread found for the current video.")

    def add_export_button(self):
        self.export_button = QPushButton("Export DataFrame", self)
        self.export_button.setGeometry(10, 480, 780, 30)
        self.export_button.clicked.connect(self.export_dataframe)
        self.export_button.setEnabled(False)  # The button is initially disabled

    def enable_export_button(self):
        self.export_button.setEnabled(True)

    # Method to dynamically add labels for new ROIs
    def add_center_mating_duration_label(self, roi_id):
        label = QLabel(f"ROI {roi_id}: Center Mating Duration: Not Available")
        label.setWordWrap(True)
        self.scroll_layout_for_center_mating_duration.addWidget(label)
        self.center_mating_duration_labels.append(label)

    def update_center_mating_duration(self, roi_id, duration):
        while len(self.center_mating_duration_labels) <= roi_id:
            self.add_center_mating_duration_label(len(self.center_mating_duration_labels))

        self.center_mating_duration_labels[roi_id].setText(
            f"ROI {roi_id}: Center Mating Duration: {duration:.2f} seconds")

    def add_center_gender_duration_label(self, roi_id, gender):
        key = (roi_id, gender)
        if key not in self.center_gender_duration_labels:
            label = QLabel(f"ROI {roi_id} ({gender}): Center Gender Duration: Not Available")
            label.setWordWrap(True)
            self.scroll_layout_for_center_gating_duration.addWidget(label)
            self.center_gender_duration_labels[key] = label

    def update_center_gender_duration(self, roi_id, male_duration, female_duration):
        self.add_center_gender_duration_label(roi_id, 'male')
        self.add_center_gender_duration_label(roi_id, 'female')

        male_key = (roi_id, 'male')
        female_key = (roi_id, 'female')

        if male_key in self.center_gender_duration_labels:
            self.center_gender_duration_labels[male_key].setText(
                f"ROI {roi_id} (male): Center Gender Duration: {male_duration:.2f} seconds")

        if female_key in self.center_gender_duration_labels:
            self.center_gender_duration_labels[female_key].setText(
                f"ROI {roi_id} (female): Center Gender Duration: {female_duration:.2f} seconds")

    def export_dataframe(self):
        for video_path, video_thread in self.video_threads.items():
            if video_thread:
                # Generate the default export name based on the video file name
                default_export_name = os.path.splitext(video_path)[0] + '_analysis.csv'

                # Prepare data for export
                data = []
                num_rois = len(video_thread.initial_contours)
                for roi in range(num_rois):
                    start_time = video_thread.mating_start_times.get(roi, 'N/A')
                    start_time = 'N/A' if start_time == 'N/A' else max(0, start_time - 360)

                    durations = video_thread.mating_durations.get(roi, [])
                    longest_duration = max(durations, default=0)
                    # Find the longest durations
                    longest_duration = 0 if longest_duration < 360 else longest_duration

                    # Mating status is true if the most recent mating event lasted at least 360 seconds
                    mating_status = durations[-1] >= 360 if durations else False

                    center_mating_durations = video_thread.center_mating_duration.get(roi, [])
                    total_center_mating_duration = sum(center_mating_durations)

                    # Calculate outside center mating duration
                    outside_center_mating_duration = max(0, longest_duration - total_center_mating_duration)

                    # Get the center gender durations
                    center_male_duration = video_thread.center_gender_duration.get(roi, {}).get('male', 0)
                    center_female_duration = video_thread.center_gender_duration.get(roi, {}).get('female', 0)

                    data.append({'ROI': roi, 'Adjusted Start Time': start_time,
                                 'Longest Duration': longest_duration,
                                 'Mating Status': mating_status, 'Center-Mating Duration': total_center_mating_duration,
                                 'Male Time in Center': center_male_duration,
                                 'Female Time in Center': center_female_duration, 'Outside Center Mating Duration':
                                     outside_center_mating_duration})

                # Create DataFrame
                mating_times_df = pd.DataFrame(data)

                # Mark void ROIs as 'N/A'
                void_rois = video_thread.void_rois
                for column in ['Adjusted Start Time', 'Longest Duration', 'Mating Status']:
                    mating_times_df[column] = mating_times_df.apply(
                        lambda row: 'N/A' if void_rois.get(row['ROI'], False) else row[column], axis=1)

                # Export to CSV
                mating_times_df.to_csv(default_export_name, index=False)
                self.processing_status_label.setText(f'DataFrame for {video_path} exported successfully.')
                QMessageBox.information(self, "Success", f"DataFrame for {video_path} exported successfully.")

    def previous_video(self):
        if self.current_video_index > 0:
            self.current_video_index -= 1
            current_video_path = self.video_paths[self.current_video_index]
            if current_video_path in self.latest_frames:
                frame = self.latest_frames[current_video_path]
                mating_durations = self.latest_mating_durations[current_video_path]
                self.update_video_frame(current_video_path, frame, mating_durations)

    def next_video(self):
        if self.current_video_index < len(self.video_paths) - 1:
            self.current_video_index += 1
            current_video_path = self.video_paths[self.current_video_index]
            if current_video_path in self.latest_frames:
                frame = self.latest_frames[current_video_path]
                mating_durations = self.latest_mating_durations[current_video_path]
                self.update_video_frame(current_video_path, frame, mating_durations)

    def update_verified_mating_times(self, video_path, mating_times_dict):
        if video_path in self.video_threads and hasattr(self.video_threads[video_path], 'mating_durations'):
            adjusted_mating_times_dict = {roi_id: max(0, time - 360) for roi_id, time in mating_times_dict.items()}
            mating_times_df = pd.DataFrame(list(adjusted_mating_times_dict.items()), columns=['ROI', 'Start Time'])

            durations = self.video_threads[video_path].mating_durations
            mating_times_df['Mating Duration'] = [durations.get(roi_id, 0) for roi_id in mating_times_df['ROI']]

            self.mating_start_times_dfs[video_path] = mating_times_df

            current_video_path = self.video_paths[self.current_video_index]
            if video_path == current_video_path:  # Only update the GUI if the video_path matches the current video
                self.enable_export_button()
                mating_time_text = "\n".join(
                    [f"ROI {roi_id}: {time:.2f} seconds" for roi_id, time in adjusted_mating_times_dict.items()])
                self.verified_mating_times_label.setText(mating_time_text)
        else:
            print("Error: video_thread for the current video is None or doesn't have the required attributes")

    def set_fps_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.fps_input.setText(str(fps))

    def select_video(self):
        # Use getOpenFileNames to select multiple videos
        video_paths, _ = QFileDialog.getOpenFileNames(self, "Select Videos")
        if video_paths:
            self.video_paths.extend(video_paths)
            for video_path in video_paths:
                self.set_fps_from_video(video_path)
                self.video_threads[video_path] = None
                # Add video filename to the list widget
                self.video_list_widget.addItem(video_path.split("/")[-1])
            # Enable start button only if at least one video is selected
            self.start_button.setEnabled(len(self.video_paths) > 0)
    def start_processing(self):
        if self.video_paths and self.fps_input.text():
            self.start_button.setEnabled(False)
            self.select_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            fps = float(self.fps_input.text())
            # Retrieve the skip frames value
            skip_seconds = float(self.skip_frames_input.text()) if self.skip_frames_input.text() else 0
            skip_frames = int(skip_seconds * fps)  # Convert seconds to frames
            try:
                perf_frame_skips = int(self.frame_skip_input.text())
            except ValueError:
                perf_frame_skips = 1 # Default value if input is invalid
            for video_path in self.video_paths:
                if video_path not in self.video_threads or not self.video_threads[video_path]:
                    video_thread = VideoProcessingThread(video_path, [], fps, skip_frames, perf_frame_skips)
                    self.video_threads[video_path] = video_thread
                    # Connect signals
                    video_thread.mating_analysis_complete.connect(self.export_dataframe)
                    video_thread.center_mating_duration_signal.connect(self.update_center_mating_duration)
                    video_thread.verified_mating_start_times.connect(self.update_verified_mating_times)
                    video_thread.center_gender_duration_signal.connect(self.update_center_gender_duration)
                    video_thread.frame_info.connect(self.update_frame_info)
                    video_thread.frame_processed.connect(self.update_video_frame)
                    video_thread.frame_processed.connect(self.update_video_frame)
                    video_thread.finished.connect(self.processing_finished)
                    video_thread.void_roi_signal.connect(self.void_roi_handler)
                    video_thread.start()

            # Enable navigation buttons if there are multiple videos
            self.prev_button.setEnabled(len(self.video_paths) > 1)
            self.next_button.setEnabled(len(self.video_paths) > 1)

    def stop_processing(self):
        # Stop all video threads
        for video_thread in self.video_threads.values():
            if video_thread and video_thread.is_running:
                video_thread.stop()
        # Update the status label to indicate that processing has stopped
        self.processing_status_label.setText('Video processing stopped.')
        # Re-enable the start and select buttons
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def processing_finished(self):
        self.processing_status_label.setText('Video processing finished.')
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_video_frame(self, video_path, frame, mating_durations):
        # Check if the frame is from the current video being displayed
        current_video_path = self.video_paths[self.current_video_index]
        if video_path == current_video_path:
            # Update video_label
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(),
                                   Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)

            # Update mating_duration_label with the newest duration for each ROI
            mating_duration_text = ""
            for roi_id, durations in mating_durations.items():
                latest_duration = durations[-1] if durations else 0
                mating_duration_text += f"ROI {roi_id}: {latest_duration:.2f} seconds\n"
            self.mating_duration_label.setText(mating_duration_text)

            # Display mating start times for the current video
            if video_path in self.mating_start_times_dfs:
                mating_times_df = self.mating_start_times_dfs[video_path]
                mating_time_text = "\n".join(
                    [f"ROI {row['ROI']}: {row['Start Time']:.2f} seconds" for _, row in mating_times_df.iterrows()])
                self.verified_mating_times_label.setText(mating_time_text)
            else:
                self.verified_mating_times_label.setText("")

        # Store the frame and mating durations for this video
        self.latest_frames[video_path] = frame
        self.latest_mating_durations[video_path] = mating_durations

    def update_frame_info(self, frame, time):
        self.frame_label.setText(f'Frame: {frame}')
        self.time_label.setText(f'Time (s): {time:.2f}')

    def void_roi_handler(self, video_path, roi_id):
        # Handle the void ROI, perhaps by updating the UI or logging
        print(f"ROI {roi_id} in video {video_path} has been marked as void.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())