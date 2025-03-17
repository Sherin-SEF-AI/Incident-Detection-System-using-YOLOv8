import sys
import os
import cv2
import numpy as np
import torch
import pandas as pd
import datetime
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QWidget,
    QProgressBar,
    QComboBox,
    QSlider,
    QGroupBox,
    QTabWidget,
    QSplitter,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QLineEdit,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QScrollArea,
    QMenu,
    QToolBar,
    QStatusBar,
    QMessageBox,
    QDockWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QUrl, QRect, QPoint
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QFont,
    QIcon,
    QPalette,
    QColor,
    QPainter,
    QPen,
    QAction,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from ultralytics import YOLO
from datetime import timedelta
import seaborn as sns
import webbrowser
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Image,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet


class VideoPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # Video display
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_frame.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.video_frame.setText("Load a video to begin analysis")
        self.video_frame.setMinimumSize(640, 480)

        # Controls layout
        self.controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_btn = QPushButton()
        self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setToolTip("Play/Pause")
        self.play_btn.clicked.connect(self.toggle_playback)

        # Previous frame button
        self.prev_frame_btn = QPushButton()
        self.prev_frame_btn.setIcon(QIcon.fromTheme("media-skip-backward"))
        self.prev_frame_btn.setFixedSize(40, 40)
        self.prev_frame_btn.setToolTip("Previous Frame")
        self.prev_frame_btn.clicked.connect(self.prev_frame)

        # Next frame button
        self.next_frame_btn = QPushButton()
        self.next_frame_btn.setIcon(QIcon.fromTheme("media-skip-forward"))
        self.next_frame_btn.setFixedSize(40, 40)
        self.next_frame_btn.setToolTip("Next Frame")
        self.next_frame_btn.clicked.connect(self.next_frame)

        # Position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setToolTip("Video Position")
        self.position_slider.sliderMoved.connect(self.set_position)

        # Current time / total time label
        self.time_label = QLabel("00:00 / 00:00")

        # Speed control
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentIndex(2)  # Default to 1.0x
        self.speed_combo.setToolTip("Playback Speed")
        self.speed_combo.currentIndexChanged.connect(self.change_speed)

        # Add widgets to controls layout
        self.controls_layout.addWidget(self.prev_frame_btn)
        self.controls_layout.addWidget(self.play_btn)
        self.controls_layout.addWidget(self.next_frame_btn)
        self.controls_layout.addWidget(self.position_slider)
        self.controls_layout.addWidget(self.time_label)
        self.controls_layout.addWidget(self.speed_combo)

        # Add everything to the main layout
        self.layout.addWidget(self.video_frame)
        self.layout.addLayout(self.controls_layout)

        # Playback variables
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.is_playing = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 0
        self.speed_factor = 1.0

        # Enable/disable controls
        self.set_controls_enabled(False)

    def load_video(self, video_path):
        if not video_path or not os.path.exists(video_path):
            return False

        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file.")
            return False

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.position_slider.setRange(0, self.total_frames - 1)
        self.current_frame_idx = 0

        # Update duration label
        duration = self.total_frames / self.fps
        self.time_label.setText(f"00:00 / {timedelta(seconds=int(duration))}")

        # Read first frame
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)

        # Enable controls
        self.set_controls_enabled(True)
        return True

    def display_frame(self, frame, detections=None):
        if frame is None:
            return

        # If we have detections, draw them
        if detections is not None:
            frame = self.draw_detections(frame.copy(), detections)

        # Convert frame to QPixmap and display
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        # Convert to RGB for Qt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)

        # Get label size
        label_size = self.video_frame.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)

        self.video_frame.setPixmap(scaled_pixmap)

    def draw_detections(self, frame, detections):
        # If detections is a YOLOv8 result, use its plot function
        if (
            hasattr(detections, "__iter__")
            and len(detections) > 0
            and hasattr(detections[0], "plot")
        ):
            annotated_frame = detections[0].plot()
            return annotated_frame

        # Otherwise, it might be our custom detection format
        # Example: draw bounding boxes for people
        if isinstance(detections, dict):
            for det_type, instances in detections.items():
                color = (0, 255, 0)  # Green for falls
                if det_type == "attacks":
                    color = (0, 0, 255)  # Red for attacks
                elif det_type == "accidents":
                    color = (255, 0, 0)  # Blue for accidents

                for instance in instances:
                    if isinstance(instance, tuple) and len(instance) >= 2:
                        # Expected format: (frame_num, details)
                        details = instance[1]
                        if "bbox" in details:
                            x1, y1, x2, y2 = details["bbox"]
                            cv2.rectangle(
                                frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                            )
                            cv2.putText(
                                frame,
                                det_type,
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                color,
                                2,
                            )

        return frame

    def toggle_playback(self):
        if not self.cap:
            return

        if self.is_playing:
            self.timer.stop()
            self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        else:
            interval = int(1000 / (self.fps * self.speed_factor))
            self.timer.start(max(1, interval))
            self.play_btn.setIcon(QIcon.fromTheme("media-playback-pause"))

        self.is_playing = not self.is_playing

    def update_frame(self):
        if not self.cap:
            return

        # Check if we've reached the end
        if self.current_frame_idx >= self.total_frames - 1:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
            return

        # Increment frame index
        self.current_frame_idx += 1

        # Set position in video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        # Read frame
        ret, frame = self.cap.read()
        if ret:
            # Update UI
            self.display_frame(frame)
            self.position_slider.setValue(self.current_frame_idx)

            # Update time label
            current_time = self.current_frame_idx / self.fps
            total_time = self.total_frames / self.fps
            self.time_label.setText(
                f"{timedelta(seconds=int(current_time))} / {timedelta(seconds=int(total_time))}"
            )

    def next_frame(self):
        if not self.cap or self.current_frame_idx >= self.total_frames - 1:
            return

        self.current_frame_idx += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.position_slider.setValue(self.current_frame_idx)

            # Update time label
            current_time = self.current_frame_idx / self.fps
            total_time = self.total_frames / self.fps
            self.time_label.setText(
                f"{timedelta(seconds=int(current_time))} / {timedelta(seconds=int(total_time))}"
            )

    def prev_frame(self):
        if not self.cap or self.current_frame_idx <= 0:
            return

        self.current_frame_idx -= 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.position_slider.setValue(self.current_frame_idx)

            # Update time label
            current_time = self.current_frame_idx / self.fps
            total_time = self.total_frames / self.fps
            self.time_label.setText(
                f"{timedelta(seconds=int(current_time))} / {timedelta(seconds=int(total_time))}"
            )

    def set_position(self, position):
        if not self.cap:
            return

        self.current_frame_idx = position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)

            # Update time label
            current_time = position / self.fps
            total_time = self.total_frames / self.fps
            self.time_label.setText(
                f"{timedelta(seconds=int(current_time))} / {timedelta(seconds=int(total_time))}"
            )

    def change_speed(self, index):
        speeds = [0.25, 0.5, 1.0, 1.5, 2.0]
        self.speed_factor = speeds[index]

        # Update timer interval if playing
        if self.is_playing:
            self.timer.stop()
            interval = int(1000 / (self.fps * self.speed_factor))
            self.timer.start(max(1, interval))

    def set_controls_enabled(self, enabled):
        self.play_btn.setEnabled(enabled)
        self.prev_frame_btn.setEnabled(enabled)
        self.next_frame_btn.setEnabled(enabled)
        self.position_slider.setEnabled(enabled)
        self.speed_combo.setEnabled(enabled)

    def update_frame_with_detection(self, frame, detections):
        self.display_frame(frame, detections)


class RegionOfInterestDialog(QDialog):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Region of Interest")
        self.frame = frame
        self.roi = None
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()

        # Set up UI
        layout = QVBoxLayout()

        self.instructions = QLabel(
            "Click and drag to select a region of interest.\n"
            "Press OK to confirm, or Cancel to use the entire frame."
        )
        layout.addWidget(self.instructions)

        self.frame_label = QLabel()
        self.frame_label.setMinimumSize(640, 480)
        layout.addWidget(self.frame_label)

        # Display initial frame
        self.display_frame()

        # Buttons
        button_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_selection)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def display_frame(self):
        frame_copy = self.frame.copy()

        # Draw current selection
        if self.start_point and self.end_point:
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert frame to QPixmap
        h, w, ch = frame_copy.shape
        bytes_per_line = ch * w
        rgb_image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)

        # Resize to fit label
        label_size = self.frame_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)

        self.frame_label.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Get position relative to label
            pos = self.frame_label.mapFromParent(event.pos())
            if self.frame_label.rect().contains(pos):
                self.drawing = True
                # Adjust coordinates based on scaled image
                pixmap = self.frame_label.pixmap()
                pixmap_rect = QRect(
                    (self.frame_label.width() - pixmap.width()) // 2,
                    (self.frame_label.height() - pixmap.height()) // 2,
                    pixmap.width(),
                    pixmap.height(),
                )

                if pixmap_rect.contains(pos):
                    # Convert to original image coordinates
                    x_ratio = self.frame.shape[1] / pixmap.width()
                    y_ratio = self.frame.shape[0] / pixmap.height()

                    x = int((pos.x() - pixmap_rect.x()) * x_ratio)
                    y = int((pos.y() - pixmap_rect.y()) * y_ratio)

                    self.start_point = QPoint(x, y)
                    self.end_point = QPoint(x, y)
                    self.display_frame()

    def mouseMoveEvent(self, event):
        if self.drawing:
            # Get position relative to label
            pos = self.frame_label.mapFromParent(event.pos())

            # Adjust coordinates based on scaled image
            pixmap = self.frame_label.pixmap()
            pixmap_rect = QRect(
                (self.frame_label.width() - pixmap.width()) // 2,
                (self.frame_label.height() - pixmap.height()) // 2,
                pixmap.width(),
                pixmap.height(),
            )

            if pixmap_rect.contains(pos):
                # Convert to original image coordinates
                x_ratio = self.frame.shape[1] / pixmap.width()
                y_ratio = self.frame.shape[0] / pixmap.height()

                x = int((pos.x() - pixmap_rect.x()) * x_ratio)
                y = int((pos.y() - pixmap_rect.y()) * y_ratio)

                self.end_point = QPoint(x, y)
                self.display_frame()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False

            # Normalize coordinates (ensure start is top-left, end is bottom-right)
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())

            self.start_point = QPoint(x1, y1)
            self.end_point = QPoint(x2, y2)

            # Store ROI
            self.roi = (x1, y1, x2, y2)

            # Enable OK button if we have a valid ROI
            if x2 > x1 and y2 > y1:
                self.ok_button.setEnabled(True)

            self.display_frame()

    def reset_selection(self):
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.roi = None
        self.ok_button.setEnabled(False)
        self.display_frame()


class CustomDetectionRuleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Detection Rule")
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Rule name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Rule Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("E.g., Crawling person")
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)

        # Rule type
        type_group = QGroupBox("Detection Type")
        type_layout = QVBoxLayout()

        self.type_buttons = QButtonGroup(self)
        self.fall_radio = QRadioButton("Fall")
        self.attack_radio = QRadioButton("Attack")
        self.accident_radio = QRadioButton("Accident")
        self.custom_radio = QRadioButton("Custom")

        self.type_buttons.addButton(self.fall_radio, 1)
        self.type_buttons.addButton(self.attack_radio, 2)
        self.type_buttons.addButton(self.accident_radio, 3)
        self.type_buttons.addButton(self.custom_radio, 4)
        self.fall_radio.setChecked(True)

        type_layout.addWidget(self.fall_radio)
        type_layout.addWidget(self.attack_radio)
        type_layout.addWidget(self.accident_radio)
        type_layout.addWidget(self.custom_radio)

        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Detection parameters
        params_group = QGroupBox("Detection Parameters")
        params_layout = QGridLayout()

        # Confidence threshold
        params_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.1, 1.0)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setValue(0.5)
        params_layout.addWidget(self.conf_threshold, 0, 1)

        # Time threshold
        params_layout.addWidget(QLabel("Minimum Duration (frames):"), 1, 0)
        self.time_threshold = QSpinBox()
        self.time_threshold.setRange(1, 100)
        self.time_threshold.setValue(5)
        params_layout.addWidget(self.time_threshold, 1, 1)

        # Person count
        params_layout.addWidget(QLabel("Minimum People Count:"), 2, 0)
        self.person_count = QSpinBox()
        self.person_count.setRange(1, 10)
        self.person_count.setValue(1)
        params_layout.addWidget(self.person_count, 2, 1)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Advanced conditions
        advanced_group = QGroupBox("Advanced Conditions")
        advanced_layout = QVBoxLayout()

        advanced_layout.addWidget(QLabel("Custom Condition Logic:"))
        self.condition_edit = QTextEdit()
        self.condition_edit.setPlaceholderText(
            "Example conditions:\n"
            "- Head position < 0.4 * body height\n"
            "- Distance between people < 100 pixels\n"
            "- Person area decreased by > 40%"
        )
        self.condition_edit.setMaximumHeight(100)
        advanced_layout.addWidget(self.condition_edit)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Rule")
        self.save_btn.clicked.connect(self.accept)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_rule_data(self):
        rule_type = "fall"
        if self.attack_radio.isChecked():
            rule_type = "attack"
        elif self.accident_radio.isChecked():
            rule_type = "accident"
        elif self.custom_radio.isChecked():
            rule_type = "custom"

        return {
            "name": self.name_edit.text(),
            "type": rule_type,
            "confidence_threshold": self.conf_threshold.value(),
            "time_threshold": self.time_threshold.value(),
            "person_count": self.person_count.value(),
            "custom_conditions": self.condition_edit.toPlainText(),
        }


class VideoProcessingThread(QThread):
    update_frame = pyqtSignal(np.ndarray, list)
    update_progress = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)

    def __init__(
        self, video_path, model, conf_threshold=0.5, roi=None, custom_rules=None
    ):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.conf_threshold = conf_threshold
        self.roi = roi  # Region of interest (x1, y1, x2, y2)
        self.custom_rules = custom_rules or []
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        incidents = {"falls": [], "attacks": [], "accidents": []}

        # Add custom rule types if any
        for rule in self.custom_rules:
            if rule["type"] not in incidents and rule["type"] != "custom":
                incidents[rule["type"]] = []

        # Keep track of recent frames for temporal analysis
        recent_frames = []
        max_recent_frames = 10

        # Tracks for people
        tracks = {}

        frame_count = 0

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply ROI if specified
            if self.roi:
                x1, y1, x2, y2 = self.roi
                frame_roi = frame[y1:y2, x1:x2]
                # Run YOLOv8 inference on ROI
                results = self.model(frame_roi, verbose=False, conf=self.conf_threshold)

                # Adjust bounding box coordinates back to original frame
                if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    boxes[:, 0] += x1
                    boxes[:, 1] += y1
                    boxes[:, 2] += x1
                    boxes[:, 3] += y1
                    # Now we need to update the results with adjusted boxes...
                    # This is simplified - would need proper implementation
            else:
                # Run YOLOv8 inference on full frame
                results = self.model(frame, verbose=False, conf=self.conf_threshold)

            # Add current frame to recent frames list
            if len(recent_frames) >= max_recent_frames:
                recent_frames.pop(0)
            recent_frames.append((frame, results))

            # Process results to detect anomalies
            detections = self.process_detections(
                results, frame_count, fps, recent_frames, tracks
            )

            # Update tracks
            self.update_tracks(results, frame_count, tracks)

            # Update incidents dictionary
            for key in incidents:
                if key in detections and detections[key]:
                    # Add bounding box to detection
                    if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                        for det in detections[key]:
                            # Match detection to a box
                            # Simplified - would need proper matching logic
                            box = results[0].boxes.xyxy[0].cpu().numpy()
                            detections[key][0]["bbox"] = box

                    incidents[key].append((frame_count, detections[key]))

            # Apply any custom rules
            for rule in self.custom_rules:
                detections = self.apply_custom_rule(
                    rule, results, frame_count, fps, recent_frames, tracks
                )
                if detections:
                    rule_type = rule["type"]
                    if rule_type == "custom":
                        rule_type = rule["name"].lower().replace(" ", "_")
                        if rule_type not in incidents:
                            incidents[rule_type] = []

                    incidents[rule_type].append((frame_count, detections))

            # Emit signals for UI updates
            self.update_frame.emit(frame, results)
            progress = int((frame_count / total_frames) * 100)
            self.update_progress.emit(progress)

            frame_count += 1

        cap.release()
        self.processing_finished.emit(incidents)

    def update_tracks(self, results, frame_count, tracks):
        # Simple tracking implementation
        # In a real application, would use a more sophisticated tracker

        if not hasattr(results[0], "boxes") or len(results[0].boxes) == 0:
            return

        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Match current detections to existing tracks
        matched_tracks = set()

        for i, box in enumerate(boxes):
            best_match = None
            best_iou = 0

            for track_id, track_info in tracks.items():
                last_box = track_info["boxes"][-1]
                iou = self.calculate_iou(box, last_box)

                if iou > 0.3 and iou > best_iou:  # IOU threshold
                    best_match = track_id
                    best_iou = iou

            if best_match is not None:
                # Update existing track
                tracks[best_match]["boxes"].append(box)
                tracks[best_match]["frames"].append(frame_count)
                matched_tracks.add(best_match)
            else:
                # Create new track
                new_id = max(tracks.keys(), default=0) + 1
                tracks[new_id] = {"boxes": [box], "frames": [frame_count]}

        # Remove old tracks (not matched for several frames)
        track_ids = list(tracks.keys())
        for track_id in track_ids:
            if track_id not in matched_tracks:
                if (
                    frame_count - tracks[track_id]["frames"][-1] > 10
                ):  # Remove after 10 frames of absence
                    del tracks[track_id]

    def calculate_iou(self, box1, box2):
        # Calculate intersection over union between two boxes
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # Check if boxes overlap
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / union

        return iou

    def process_detections(self, results, frame_count, fps, recent_frames, tracks):
        # This is a simplified detection logic
        # In a real application, you would implement more sophisticated algorithms
        detections = {}

        # Example: Analyze poses to detect falls
        # This requires pose estimation from YOLOv8 pose model
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            poses = results[0].keypoints.data
            for i, pose in enumerate(poses):
                keypoints = pose.cpu().numpy()

                if self.detect_fall(keypoints):
                    confidence = 0.8
                    detections["falls"] = [
                        {
                            "confidence": confidence,
                            "time": frame_count / fps,
                            "person_idx": i,
                        }
                    ]

        # Look for violent actions (simplified)
        boxes = results[0].boxes if hasattr(results[0], "boxes") else None
        if boxes is not None and len(boxes) >= 2:  # Multiple people detected
            if self.detect_attack(boxes):
                detections["attacks"] = [{"confidence": 0.7, "time": frame_count / fps}]

        # Detect accidents (simplified logic)
        if len(recent_frames) >= 3:
            if self.detect_accident(recent_frames, frame_count, fps):
                detections["accidents"] = [
                    {"confidence": 0.6, "time": frame_count / fps}
                ]

        return detections

    def detect_fall(self, pose_keypoints):
        # Fall detection logic
        # Check if head keypoint is close to ground level
        if len(pose_keypoints) >= 17:  # COCO keypoints format
            nose = pose_keypoints[0]  # Nose keypoint
            left_ankle = pose_keypoints[15]  # Left ankle keypoint
            right_ankle = pose_keypoints[16]  # Right ankle keypoint

            # Check if keypoints are valid (visible)
            if nose[2] > 0.5 and (left_ankle[2] > 0.5 or right_ankle[2] > 0.5):
                head_y = nose[1]
                ankle_y = (
                    max(left_ankle[1], right_ankle[1])
                    if left_ankle[2] > 0.5 and right_ankle[2] > 0.5
                    else left_ankle[1] if left_ankle[2] > 0.5 else right_ankle[1]
                )

                # Check if head is close to ground level (near ankles)
                if head_y > ankle_y * 0.8:

                    # Check body orientation (horizontal)
                    # Get shoulder keypoints
                    left_shoulder = pose_keypoints[5]
                    right_shoulder = pose_keypoints[6]

                    if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                        shoulder_diff_y = abs(left_shoulder[1] - right_shoulder[1])
                        shoulder_diff_x = abs(left_shoulder[0] - right_shoulder[0])

                        # If shoulders are more horizontal than vertical
                        if shoulder_diff_y < shoulder_diff_x:
                            return True

        return False

    def detect_attack(self, boxes):
        # Attack detection
        if not hasattr(boxes, "xyxy") or len(boxes.xyxy) < 2:
            return False

        box_data = boxes.xyxy.cpu().numpy()

        # Check for close proximity between people
        for i in range(len(box_data)):
            for j in range(i + 1, len(box_data)):
                box1 = box_data[i]
                box2 = box_data[j]

                # Calculate centers
                center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
                center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

                # Calculate distance between centers
                distance = np.sqrt(
                    (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
                )

                # Calculate average box size for reference
                avg_width = (box1[2] - box1[0] + box2[2] - box2[0]) / 2

                # If people are very close (distance less than average width)
                if distance < avg_width * 0.8:
                    return True

        return False

    def detect_accident(self, recent_frames, frame_count, fps):
        # Simplified accident detection
        # Looking for sudden movements or changes in position

        if len(recent_frames) < 3:
            return False

        # Get current and previous frames
        current_frame, current_results = recent_frames[-1]
        prev_frame, prev_results = recent_frames[
            -3
        ]  # Skip one frame to see more pronounced changes

        # Check if we have detections in both frames
        if not hasattr(current_results[0], "boxes") or not hasattr(
            prev_results[0], "boxes"
        ):
            return False

        current_boxes = (
            current_results[0].boxes.xyxy.cpu().numpy()
            if len(current_results[0].boxes) > 0
            else []
        )
        prev_boxes = (
            prev_results[0].boxes.xyxy.cpu().numpy()
            if len(prev_results[0].boxes) > 0
            else []
        )

        if len(current_boxes) == 0 or len(prev_boxes) == 0:
            return False

        # Calculate motion between frames
        max_motion = 0

        for i, curr_box in enumerate(current_boxes):
            # Find closest box in previous frame
            best_iou = 0
            best_motion = 0

            for prev_box in prev_boxes:
                iou = self.calculate_iou(curr_box, prev_box)

                if iou > 0.3:  # Assume it's the same person if IoU > 0.3
                    # Calculate center points
                    curr_center = (
                        (curr_box[0] + curr_box[2]) / 2,
                        (curr_box[1] + curr_box[3]) / 2,
                    )
                    prev_center = (
                        (prev_box[0] + prev_box[2]) / 2,
                        (prev_box[1] + prev_box[3]) / 2,
                    )

                    # Calculate motion (displacement)
                    motion = np.sqrt(
                        (curr_center[0] - prev_center[0]) ** 2
                        + (curr_center[1] - prev_center[1]) ** 2
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_motion = motion

            if best_iou > 0 and best_motion > max_motion:
                max_motion = best_motion

        # Consider as accident if motion exceeds threshold
        # This threshold would need tuning for real applications
        motion_threshold = 50  # pixels

        return max_motion > motion_threshold

    def apply_custom_rule(self, rule, results, frame_count, fps, recent_frames, tracks):
        # Apply a custom detection rule
        rule_type = rule["type"]
        conf_threshold = rule["confidence_threshold"]
        person_count = rule["person_count"]

        # Basic checks
        if not hasattr(results[0], "boxes") or len(results[0].boxes) < person_count:
            return None

        # Check rule type
        if rule_type == "fall":
            # Enhanced fall detection using custom parameters
            if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
                poses = results[0].keypoints.data
                for i, pose in enumerate(poses):
                    keypoints = pose.cpu().numpy()

                    if self.detect_fall(keypoints):
                        # Apply custom confidence threshold
                        confidence = 0.8
                        if confidence >= conf_threshold:
                            return [
                                {
                                    "confidence": confidence,
                                    "time": frame_count / fps,
                                    "person_idx": i,
                                    "rule": rule["name"],
                                }
                            ]

        elif rule_type == "attack":
            # Enhanced attack detection
            if self.detect_attack(results[0].boxes):
                confidence = 0.7
                if confidence >= conf_threshold:
                    return [
                        {
                            "confidence": confidence,
                            "time": frame_count / fps,
                            "rule": rule["name"],
                        }
                    ]

        elif rule_type == "accident":
            # Enhanced accident detection
            if len(recent_frames) >= 3:
                if self.detect_accident(recent_frames, frame_count, fps):
                    confidence = 0.6
                    if confidence >= conf_threshold:
                        return [
                            {
                                "confidence": confidence,
                                "time": frame_count / fps,
                                "rule": rule["name"],
                            }
                        ]

        elif rule_type == "custom":
            # Process custom conditions - this would need to parse the custom logic
            # This is just a placeholder implementation
            # In a real application, you would interpret the custom conditions

            # Example: detect people staying in one place too long
            if len(tracks) > 0:
                for track_id, track_info in tracks.items():
                    if len(track_info["frames"]) > rule["time_threshold"]:
                        # Check if person hasn't moved much
                        if len(track_info["boxes"]) >= 2:
                            first_box = track_info["boxes"][0]
                            last_box = track_info["boxes"][-1]

                            # Calculate centers
                            first_center = (
                                (first_box[0] + first_box[2]) / 2,
                                (first_box[1] + first_box[3]) / 2,
                            )
                            last_center = (
                                (last_box[0] + last_box[2]) / 2,
                                (last_box[1] + last_box[3]) / 2,
                            )

                            # Calculate movement
                            movement = np.sqrt(
                                (last_center[0] - first_center[0]) ** 2
                                + (last_center[1] - first_center[1]) ** 2
                            )

                            # If movement is small, consider it a match for this rule
                            if movement < 30:  # Threshold for "stationary"
                                return [
                                    {
                                        "confidence": 0.7,
                                        "time": frame_count / fps,
                                        "track_id": track_id,
                                        "rule": rule["name"],
                                    }
                                ]

        return None

    def stop(self):
        self.running = False


class IncidentDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Incident Detection System")
        self.setMinimumSize(1200, 800)

        # Initialize YOLO model
        self.model = None
        self.load_model()

        # Setup UI
        self.init_ui()

        # Variables
        self.video_path = None
        self.current_frame = None
        self.processing_thread = None
        self.incidents = {}
        self.roi = None
        self.custom_rules = []
        self.batch_file_list = []

    def load_model(self):
        try:
            # Using YOLOv8 pose model for human pose estimation
            self.model = YOLO("yolov8n-pose.pt")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")

    def init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create toolbar
        self.create_toolbar()

        # Top bar with controls
        top_bar = QHBoxLayout()

        # Load video button
        self.load_btn = QPushButton("Load Video")
        self.load_btn.setFixedSize(150, 40)
        self.load_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_btn.clicked.connect(self.load_video)
        top_bar.addWidget(self.load_btn)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8n-pose", "YOLOv8s-pose", "YOLOv8m-pose"])
        self.model_combo.setFixedWidth(200)
        self.model_combo.currentIndexChanged.connect(self.change_model)

        model_layout.addWidget(QLabel("YOLO Model:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        model_group.setFixedWidth(350)
        top_bar.addWidget(model_group)

        # Confidence threshold
        threshold_group = QGroupBox("Detection Settings")
        threshold_layout = QHBoxLayout()

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 10)
        self.threshold_slider.setValue(5)
        self.threshold_slider.setFixedWidth(150)

        self.threshold_label = QLabel("Confidence: 0.5")
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        threshold_layout.addWidget(QLabel("Threshold:"))
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        threshold_group.setLayout(threshold_layout)
        threshold_group.setFixedWidth(350)
        top_bar.addWidget(threshold_group)

        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.setFixedSize(150, 40)
        self.process_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_video)
        top_bar.addWidget(self.process_btn)

        main_layout.addLayout(top_bar)

        # Main content area with splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Video player widget
        self.video_player = VideoPlayerWidget()
        content_splitter.addWidget(self.video_player)

        # Results and visualization tabs
        self.results_tabs = QTabWidget()

        # Incidents tab
        self.incidents_tab = QWidget()
        incidents_layout = QVBoxLayout()
        self.incidents_tab.setLayout(incidents_layout)

        # Incidents table
        self.incidents_table = QTableWidget()
        self.incidents_table.setColumnCount(5)
        self.incidents_table.setHorizontalHeaderLabels(
            ["Type", "Time", "Confidence", "Duration", "Actions"]
        )
        self.incidents_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        incidents_layout.addWidget(self.incidents_table)

        # Filter controls
        filter_group = QGroupBox("Filter Incidents")
        filter_layout = QHBoxLayout()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            ["All Types", "Falls", "Attacks", "Accidents", "Custom"]
        )
        self.filter_combo.currentIndexChanged.connect(self.filter_incidents)

        self.min_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_conf_slider.setRange(1, 10)
        self.min_conf_slider.setValue(3)
        self.min_conf_slider.valueChanged.connect(self.filter_incidents)

        self.min_conf_label = QLabel("Min Confidence: 0.3")
        self.min_conf_slider.valueChanged.connect(self.update_min_conf_label)

        filter_layout.addWidget(QLabel("Type:"))
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addWidget(QLabel("Min Confidence:"))
        filter_layout.addWidget(self.min_conf_slider)
        filter_layout.addWidget(self.min_conf_label)

        filter_group.setLayout(filter_layout)
        incidents_layout.addWidget(filter_group)

        self.results_tabs.addTab(self.incidents_tab, "Incidents")

        # Visualization tab
        self.viz_tab = QWidget()
        viz_layout = QVBoxLayout()
        self.viz_tab.setLayout(viz_layout)

        # Visualization type selector
        viz_controls = QHBoxLayout()
        viz_controls.addWidget(QLabel("Visualization Type:"))

        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems(
            ["Timeline", "Heatmap", "Frequency", "Distribution"]
        )
        self.viz_type_combo.currentIndexChanged.connect(self.update_visualization)
        viz_controls.addWidget(self.viz_type_combo)

        viz_layout.addLayout(viz_controls)

        # Matplotlib canvas for visualizations
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        viz_layout.addWidget(self.canvas)

        self.results_tabs.addTab(self.viz_tab, "Visualization")

        # Custom Rules tab
        self.rules_tab = QWidget()
        rules_layout = QVBoxLayout()
        self.rules_tab.setLayout(rules_layout)

        rules_buttons = QHBoxLayout()

        self.add_rule_btn = QPushButton("Add Rule")
        self.add_rule_btn.clicked.connect(self.add_custom_rule)
        rules_buttons.addWidget(self.add_rule_btn)

        self.edit_rule_btn = QPushButton("Edit Rule")
        self.edit_rule_btn.setEnabled(False)
        self.edit_rule_btn.clicked.connect(self.edit_custom_rule)
        rules_buttons.addWidget(self.edit_rule_btn)

        self.remove_rule_btn = QPushButton("Remove Rule")
        self.remove_rule_btn.setEnabled(False)
        self.remove_rule_btn.clicked.connect(self.remove_custom_rule)
        rules_buttons.addWidget(self.remove_rule_btn)

        rules_layout.addLayout(rules_buttons)

        # Rules list
        self.rules_list = QListWidget()
        self.rules_list.itemSelectionChanged.connect(self.on_rule_selection_changed)
        rules_layout.addWidget(self.rules_list)

        self.results_tabs.addTab(self.rules_tab, "Custom Rules")

        # Batch Processing tab
        self.batch_tab = QWidget()
        batch_layout = QVBoxLayout()
        self.batch_tab.setLayout(batch_layout)

        batch_buttons = QHBoxLayout()

        self.add_batch_btn = QPushButton("Add Videos")
        self.add_batch_btn.clicked.connect(self.add_batch_videos)
        batch_buttons.addWidget(self.add_batch_btn)

        self.remove_batch_btn = QPushButton("Remove Selected")
        self.remove_batch_btn.setEnabled(False)
        self.remove_batch_btn.clicked.connect(self.remove_batch_video)
        batch_buttons.addWidget(self.remove_batch_btn)

        self.process_batch_btn = QPushButton("Process All")
        self.process_batch_btn.setEnabled(False)
        self.process_batch_btn.clicked.connect(self.process_batch)
        batch_buttons.addWidget(self.process_batch_btn)

        batch_layout.addLayout(batch_buttons)

        # Batch file list
        self.batch_list = QListWidget()
        self.batch_list.itemSelectionChanged.connect(self.on_batch_selection_changed)
        batch_layout.addWidget(self.batch_list)

        # Batch progress
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        batch_layout.addWidget(self.batch_progress)

        self.results_tabs.addTab(self.batch_tab, "Batch Processing")

        content_splitter.addWidget(self.results_tabs)
        content_splitter.setSizes([600, 400])  # Set initial sizes

        main_layout.addWidget(content_splitter)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        main_layout.addWidget(self.progress_bar)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Export action
        export_action = QAction(
            QIcon.fromTheme("document-save"), "Export Results", self
        )
        export_action.setStatusTip("Export detection results")
        export_action.triggered.connect(self.export_results)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        # ROI action
        roi_action = QAction(QIcon.fromTheme("transform-crop"), "Select ROI", self)
        roi_action.setStatusTip("Select region of interest")
        roi_action.triggered.connect(self.select_roi)
        toolbar.addAction(roi_action)

        # Clear ROI action
        clear_roi_action = QAction(QIcon.fromTheme("edit-clear"), "Clear ROI", self)
        clear_roi_action.setStatusTip("Clear region of interest")
        clear_roi_action.triggered.connect(self.clear_roi)
        toolbar.addAction(clear_roi_action)

        toolbar.addSeparator()

        # Help action
        help_action = QAction(QIcon.fromTheme("help-contents"), "Help", self)
        help_action.setStatusTip("Show help")
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)

    def load_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if video_path:
            self.video_path = video_path
            self.statusBar().showMessage(f"Loaded: {os.path.basename(video_path)}")

            # Load video into player
            if self.video_player.load_video(video_path):
                self.process_btn.setEnabled(True)

                # Reset incidents
                self.incidents = {}
                self.incidents_table.setRowCount(0)

                # Reset ROI
                self.roi = None

    def change_model(self, index):
        model_names = ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"]
        try:
            self.model = YOLO(model_names[index])
            self.statusBar().showMessage(f"Loaded model: {model_names[index]}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading model: {e}")

    def update_threshold(self):
        value = self.threshold_slider.value() / 10.0
        self.threshold_label.setText(f"Confidence: {value:.1f}")

    def update_min_conf_label(self):
        value = self.min_conf_slider.value() / 10.0
        self.min_conf_label.setText(f"Min Confidence: {value:.1f}")

    def process_video(self):
        if self.video_path is None:
            return

        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.process_btn.setText("Process Video")
            self.process_btn.setIcon(QIcon.fromTheme("media-playback-start"))
            return

        # Get confidence threshold
        conf_threshold = self.threshold_slider.value() / 10.0

        # Start processing thread
        self.processing_thread = VideoProcessingThread(
            self.video_path, self.model, conf_threshold, self.roi, self.custom_rules
        )

        # Connect signals
        self.processing_thread.update_frame.connect(self.on_frame_processed)
        self.processing_thread.update_progress.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)

        # Update UI
        self.process_btn.setText("Stop Processing")
        self.process_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.statusBar().showMessage("Processing video...")

        # Start processing
        self.processing_thread.start()

    def on_frame_processed(self, frame, results):
        self.video_player.display_frame(frame, results)

    def on_processing_finished(self, incidents):
        self.incidents = incidents
        self.process_btn.setText("Process Video")
        self.process_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.statusBar().showMessage("Processing complete")

        # Update incidents table
        self.update_incidents_table()

        # Update visualization
        self.update_visualization()

    def update_incidents_table(self):
        # Clear table
        self.incidents_table.setRowCount(0)

        # Add incidents to table
        row = 0
        for incident_type, instances in self.incidents.items():
            for frame_num, details in instances:
                # Skip if not the first detail (for simplicity)
                if not isinstance(details, list):
                    details = [details]

                for detail in details:
                    self.incidents_table.insertRow(row)

                    # Type
                    type_item = QTableWidgetItem(incident_type.capitalize())
                    self.incidents_table.setItem(row, 0, type_item)

                    # Time
                    time_value = detail.get(
                        "time", frame_num / 30.0
                    )  # Assuming 30fps if not specified
                    time_item = QTableWidgetItem(f"{time_value:.2f}s")
                    self.incidents_table.setItem(row, 1, time_item)

                    # Confidence
                    conf_value = detail.get("confidence", 0.5)
                    conf_item = QTableWidgetItem(f"{conf_value:.2f}")
                    self.incidents_table.setItem(row, 2, conf_item)

                    # Duration - placeholder
                    duration_item = QTableWidgetItem("N/A")
                    self.incidents_table.setItem(row, 3, duration_item)

                    # Actions
                    # In a real app, you'd add buttons here
                    actions_item = QTableWidgetItem("View")
                    self.incidents_table.setItem(row, 4, actions_item)

                    row += 1

    def filter_incidents(self):
        filter_type = self.filter_combo.currentText().lower()
        min_conf = self.min_conf_slider.value() / 10.0

        # Hide rows based on filter
        for row in range(self.incidents_table.rowCount()):
            type_item = self.incidents_table.item(row, 0)
            conf_item = self.incidents_table.item(row, 2)

            if not type_item or not conf_item:
                continue

            type_text = type_item.text().lower()
            conf_value = float(conf_item.text())

            # Apply filters
            show_row = True

            # Filter by type
            if filter_type != "all types" and type_text != filter_type:
                show_row = False

            # Filter by confidence
            if conf_value < min_conf:
                show_row = False

            # Show/hide row
            self.incidents_table.setRowHidden(row, not show_row)

    def update_visualization(self):
        # Clear previous plot
        self.ax.clear()

        # Get visualization type
        viz_type = self.viz_type_combo.currentText().lower()

        if viz_type == "timeline":
            self.plot_timeline()
        elif viz_type == "heatmap":
            self.plot_heatmap()
        elif viz_type == "frequency":
            self.plot_frequency()
        elif viz_type == "distribution":
            self.plot_distribution()

        # Redraw canvas
        self.canvas.draw()

    def plot_timeline(self):
        # Prepare data for timeline visualization
        times = []
        categories = []
        confidences = []

        for incident_type, incidents in self.incidents.items():
            for frame_num, details in incidents:
                if not isinstance(details, list):
                    details = [details]

                for detail in details:
                    time_sec = detail.get(
                        "time", frame_num / 30.0
                    )  # Assuming 30fps if not specified
                    times.append(time_sec)
                    categories.append(incident_type)
                    confidences.append(detail.get("confidence", 0.5) * 100)

        if not times:
            self.ax.text(
                0.5,
                0.5,
                "No incidents detected",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.ax.transAxes,
            )
        else:
            # Create scatter plot
            unique_categories = sorted(set(categories))
            category_numbers = [unique_categories.index(c) for c in categories]

            scatter = self.ax.scatter(
                times,
                category_numbers,
                s=confidences,
                alpha=0.7,
                c=confidences,
                cmap="viridis",
            )

            # Add colorbar
            cbar = self.figure.colorbar(scatter, ax=self.ax)
            cbar.set_label("Confidence (%)")

            # Set y-axis labels
            self.ax.set_yticks(range(len(unique_categories)))
            self.ax.set_yticklabels([c.capitalize() for c in unique_categories])

            # Set title and labels
            self.ax.set_title("Incident Timeline")
            self.ax.set_xlabel("Time (seconds)")
            self.ax.grid(True)

    def plot_heatmap(self):
        # Simplified heatmap - in a real application, you would create a heatmap
        # based on the actual positions of detected incidents

        if not self.incidents:
            self.ax.text(
                0.5,
                0.5,
                "No incidents detected",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.ax.transAxes,
            )
            return

        # Create a dummy heatmap (10x10 grid)
        heatmap_data = np.zeros((10, 10))

        # Populate heatmap with random incident locations
        # In a real app, you would use actual bbox coordinates
        for incident_type, incidents in self.incidents.items():
            for _ in range(len(incidents)):
                x = np.random.randint(0, 10)
                y = np.random.randint(0, 10)
                heatmap_data[y, x] += 1

        # Plot heatmap
        im = self.ax.imshow(heatmap_data, cmap="hot")

        # Add colorbar
        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.set_label("Incident Count")

        # Set title
        self.ax.set_title("Incident Location Heatmap")
        self.ax.set_xlabel("X position (video)")
        self.ax.set_ylabel("Y position (video)")

        # Remove ticks for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def plot_frequency(self):
        if not self.incidents:
            self.ax.text(
                0.5,
                0.5,
                "No incidents detected",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.ax.transAxes,
            )
            return

        # Count incidents by type
        incident_counts = {}
        for incident_type, incidents in self.incidents.items():
            incident_counts[incident_type.capitalize()] = len(incidents)

        # Create bar chart
        types = list(incident_counts.keys())
        counts = list(incident_counts.values())

        bars = self.ax.bar(types, counts, color="steelblue")

        # Add count labels above bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom",
            )

        # Set title and labels
        self.ax.set_title("Incident Frequency by Type")
        self.ax.set_xlabel("Incident Type")
        self.ax.set_ylabel("Count")

        # Rotate x-labels for better readability if needed
        self.ax.set_xticklabels(types, rotation=45, ha="right")

    def plot_distribution(self):
        if not self.incidents:
            self.ax.text(
                0.5,
                0.5,
                "No incidents detected",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.ax.transAxes,
            )
            return

        # Gather times for each incident type
        incident_times = {}
        for incident_type, incidents in self.incidents.items():
            times = []
            for frame_num, details in incidents:
                if not isinstance(details, list):
                    details = [details]

                for detail in details:
                    time_sec = detail.get("time", frame_num / 30.0)  # Assuming 30fps
                    times.append(time_sec)

            if times:
                incident_times[incident_type.capitalize()] = times

        # Create violin plot
        if incident_times:
            # Extract data for plotting
            labels = []
            data = []

            for label, times in incident_times.items():
                labels.append(label)
                data.append(times)

            # Create violin plot
            self.ax.violinplot(data, showmeans=True, showmedians=True)

            # Set x-axis labels
            self.ax.set_xticks(range(1, len(labels) + 1))
            self.ax.set_xticklabels(labels)

            # Set title and labels
            self.ax.set_title("Time Distribution of Incidents")
            self.ax.set_xlabel("Incident Type")
            self.ax.set_ylabel("Time (seconds)")

            # Add grid for readability
            self.ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    def select_roi(self):
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first.")
            return

        # Get a frame from the video
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            QMessageBox.warning(self, "Error", "Could not read frame from video.")
            return

        # Show ROI selection dialog
        roi_dialog = RegionOfInterestDialog(frame, self)
        if roi_dialog.exec() == QDialog.DialogCode.Accepted and roi_dialog.roi:
            self.roi = roi_dialog.roi
            self.statusBar().showMessage(f"ROI set: {self.roi}")

    def clear_roi(self):
        self.roi = None
        self.statusBar().showMessage("ROI cleared")

    def add_custom_rule(self):
        rule_dialog = CustomDetectionRuleDialog(self)
        if rule_dialog.exec() == QDialog.DialogCode.Accepted:
            rule_data = rule_dialog.get_rule_data()

            # Add rule to list
            self.custom_rules.append(rule_data)

            # Update list widget
            self.update_rules_list()

    def edit_custom_rule(self):
        selected_items = self.rules_list.selectedItems()
        if not selected_items:
            return

        selected_index = self.rules_list.row(selected_items[0])

        # Open dialog with current rule data
        rule_dialog = CustomDetectionRuleDialog(self)

        # Pre-fill dialog with existing rule data
        rule_data = self.custom_rules[selected_index]
        rule_dialog.name_edit.setText(rule_data["name"])

        # Set rule type
        if rule_data["type"] == "fall":
            rule_dialog.fall_radio.setChecked(True)
        elif rule_data["type"] == "attack":
            rule_dialog.attack_radio.setChecked(True)
        elif rule_data["type"] == "accident":
            rule_dialog.accident_radio.setChecked(True)
        else:
            rule_dialog.custom_radio.setChecked(True)

        # Set parameters
        rule_dialog.conf_threshold.setValue(rule_data["confidence_threshold"])
        rule_dialog.time_threshold.setValue(rule_data["time_threshold"])
        rule_dialog.person_count.setValue(rule_data["person_count"])

        # Set custom conditions
        if "custom_conditions" in rule_data:
            rule_dialog.condition_edit.setText(rule_data["custom_conditions"])

        # Show dialog
        if rule_dialog.exec() == QDialog.DialogCode.Accepted:
            # Update rule with new data
            self.custom_rules[selected_index] = rule_dialog.get_rule_data()

            # Update list widget
            self.update_rules_list()

    def remove_custom_rule(self):
        selected_items = self.rules_list.selectedItems()
        if not selected_items:
            return

        selected_index = self.rules_list.row(selected_items[0])

        # Remove rule
        self.custom_rules.pop(selected_index)

        # Update list widget
        self.update_rules_list()

    def update_rules_list(self):
        self.rules_list.clear()

        for rule in self.custom_rules:
            item_text = f"{rule['name']} ({rule['type'].capitalize()})"
            self.rules_list.addItem(item_text)

        # Update button states
        self.edit_rule_btn.setEnabled(False)
        self.remove_rule_btn.setEnabled(False)

        # Update process batch button
        self.process_batch_btn.setEnabled(len(self.batch_file_list) > 0)

    def on_rule_selection_changed(self):
        selected = len(self.rules_list.selectedItems()) > 0
        self.edit_rule_btn.setEnabled(selected)
        self.remove_rule_btn.setEnabled(selected)

    def add_batch_videos(self):
        file_dialog = QFileDialog()
        video_paths, _ = file_dialog.getOpenFileNames(
            self,
            "Select Videos for Batch Processing",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )

        if video_paths:
            # Add to list
            for path in video_paths:
                if path not in self.batch_file_list:
                    self.batch_file_list.append(path)
                    self.batch_list.addItem(os.path.basename(path))

            # Enable process button
            self.process_batch_btn.setEnabled(True)

    def remove_batch_video(self):
        selected_items = self.batch_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            index = self.batch_list.row(item)
            self.batch_file_list.pop(index)
            self.batch_list.takeItem(index)

        # Update button state
        self.process_batch_btn.setEnabled(len(self.batch_file_list) > 0)

    def on_batch_selection_changed(self):
        selected = len(self.batch_list.selectedItems()) > 0
        self.remove_batch_btn.setEnabled(selected)

    def process_batch(self):
        # This would implement batch processing
        # For brevity, just show a message
        QMessageBox.information(
            self,
            "Batch Processing",
            f"Processing {len(self.batch_file_list)} videos with current settings.",
        )

        # In a real application, you would process each video in sequence
        # and aggregate the results

    def export_results(self):
        if not self.incidents:
            QMessageBox.warning(self, "Warning", "No incidents to export.")
            return

        # Create menu with export options
        export_menu = QMenu(self)
        export_csv = export_menu.addAction("Export to CSV")
        export_pdf = export_menu.addAction("Export to PDF Report")
        export_json = export_menu.addAction("Export to JSON")

        # Show menu at cursor position
        action = export_menu.exec(QCursor.pos())

        if action == export_csv:
            self.export_to_csv()
        elif action == export_pdf:
            self.export_to_pdf()
        elif action == export_json:
            self.export_to_json()

    def export_to_csv(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save CSV Report", "", "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", newline="") as csvfile:
                # Create CSV header
                csvfile.write("Incident Type,Time (seconds),Confidence,Details\n")

                # Write incidents
                for incident_type, incidents in self.incidents.items():
                    for frame_num, details in incidents:
                        if not isinstance(details, list):
                            details = [details]

                        for detail in details:
                            time_sec = detail.get("time", frame_num / 30.0)
                            confidence = detail.get("confidence", 0.5)

                            # Format details as string
                            details_str = ",".join(
                                [
                                    f"{k}={v}"
                                    for k, v in detail.items()
                                    if k not in ["time", "confidence"]
                                ]
                            )

                            csvfile.write(
                                f'{incident_type},{time_sec:.2f},{confidence:.2f},"{details_str}"\n'
                            )

            self.statusBar().showMessage(f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error exporting to CSV: {str(e)}"
            )

    def export_to_pdf(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save PDF Report", "", "PDF Files (*.pdf)"
        )

        if not file_path:
            return

        try:
            # Create PDF document
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            elements = []

            # Add title
            styles = getSampleStyleSheet()
            elements.append(Paragraph("Incident Detection Report", styles["Title"]))
            elements.append(Spacer(1, 12))

            # Add video info
            elements.append(
                Paragraph(
                    f"Video: {os.path.basename(self.video_path)}", styles["Heading2"]
                )
            )
            elements.append(
                Paragraph(
                    f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    styles["Normal"],
                )
            )
            elements.append(Spacer(1, 12))

            # Add incidents table
            data = [["Type", "Time (s)", "Confidence", "Details"]]

            for incident_type, incidents in self.incidents.items():
                for frame_num, details in incidents:
                    if not isinstance(details, list):
                        details = [details]

                    for detail in details:
                        time_sec = detail.get("time", frame_num / 30.0)
                        confidence = detail.get("confidence", 0.5)

                        # Format details as string
                        details_str = ", ".join(
                            [
                                f"{k}={v}"
                                for k, v in detail.items()
                                if k not in ["time", "confidence"]
                            ]
                        )

                        data.append(
                            [
                                incident_type.capitalize(),
                                f"{time_sec:.2f}",
                                f"{confidence:.2f}",
                                details_str,
                            ]
                        )

            if len(data) > 1:
                table = Table(data)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )
                elements.append(table)
            else:
                elements.append(Paragraph("No incidents detected", styles["Normal"]))

            # Build document
            doc.build(elements)

            self.statusBar().showMessage(f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error exporting to PDF: {str(e)}"
            )

    def export_to_json(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save JSON Data", "", "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            # Convert incidents to serializable format
            json_data = {
                "video": os.path.basename(self.video_path),
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "incidents": {},
            }

            for incident_type, incidents in self.incidents.items():
                json_data["incidents"][incident_type] = []

                for frame_num, details in incidents:
                    if not isinstance(details, list):
                        details = [details]

                    for detail in details:
                        # Convert numpy values to Python types
                        serializable_detail = {}
                        for k, v in detail.items():
                            if isinstance(v, np.ndarray):
                                serializable_detail[k] = v.tolist()
                            elif isinstance(v, np.generic):
                                serializable_detail[k] = v.item()
                            else:
                                serializable_detail[k] = v

                        json_data["incidents"][incident_type].append(
                            {"frame": int(frame_num), "details": serializable_detail}
                        )

            # Write to file
            with open(file_path, "w") as f:
                json.dump(json_data, f, indent=2)

            self.statusBar().showMessage(f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error exporting to JSON: {str(e)}"
            )

    def show_help(self):
        help_text = """
        <h2>Incident Detection System Help</h2>
        
        <h3>Main Features:</h3>
        <ul>
            <li><b>Video Processing:</b> Load and analyze videos for incidents</li>
            <li><b>Custom Detection Rules:</b> Create your own detection criteria</li>
            <li><b>Region of Interest:</b> Focus detection on specific areas</li>
            <li><b>Batch Processing:</b> Process multiple videos at once</li>
            <li><b>Advanced Visualizations:</b> View incidents in multiple formats</li>
            <li><b>Export Results:</b> Save findings as CSV, PDF, or JSON</li>
        </ul>
        
        <h3>Quick Start Guide:</h3>
        <ol>
            <li>Load a video using the 'Load Video' button</li>
            <li>Adjust confidence threshold as needed</li>
            <li>Click 'Process Video' to start detection</li>
            <li>Review incidents in the Incidents tab</li>
            <li>Explore visualizations in the Visualization tab</li>
        </ol>
        
        <h3>Tips for Better Results:</h3>
        <ul>
            <li>Use higher confidence thresholds to reduce false positives</li>
            <li>Select a region of interest to focus on important areas</li>
            <li>Create custom rules for specific detection scenarios</li>
            <li>Try different models for varying performance levels</li>
        </ul>
        """

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Help")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(help_text)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for a cleaner look

    # Set dark theme
    dark_palette = app.palette()
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.Window, QColor(53, 53, 53)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.WindowText, QColor(255, 255, 255)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.Base, QColor(25, 25, 25)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.AlternateBase, QColor(53, 53, 53)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal,
        QPalette.ColorRole.ToolTipBase,
        QColor(255, 255, 255),
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal,
        QPalette.ColorRole.ToolTipText,
        QColor(255, 255, 255),
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.Text, QColor(255, 255, 255)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.Button, QColor(53, 53, 53)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText, QColor(255, 255, 255)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.Link, QColor(42, 130, 218)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.Highlight, QColor(42, 130, 218)
    )
    dark_palette.setColor(
        QPalette.ColorGroup.Normal, QPalette.ColorRole.HighlightedText, QColor(0, 0, 0)
    )
    app.setPalette(dark_palette)

    window = IncidentDetectionApp()
    window.show()
    sys.exit(app.exec())
