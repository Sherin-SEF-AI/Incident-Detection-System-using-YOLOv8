import sys
import os
import cv2
import numpy as np
import torch
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
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from ultralytics import YOLO


class VideoProcessingThread(QThread):
    update_frame = pyqtSignal(np.ndarray, list)
    update_progress = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)

    def __init__(self, video_path, model, conf_threshold=0.5):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.conf_threshold = conf_threshold
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        incidents = {"falls": [], "attacks": [], "accidents": []}

        frame_count = 0

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 inference on frame
            results = self.model(frame, verbose=False)

            # Process results to detect anomalies
            detections = self.process_detections(results, frame_count, fps)

            # Update incidents dictionary
            for key in incidents:
                if key in detections and detections[key]:
                    incidents[key].append((frame_count, detections[key]))

            # Emit signals for UI updates
            self.update_frame.emit(frame, results)
            progress = int((frame_count / total_frames) * 100)
            self.update_progress.emit(progress)

            frame_count += 1

        cap.release()
        self.processing_finished.emit(incidents)

    def process_detections(self, results, frame_count, fps):
        # This is a simplified detection logic
        # In a real application, you would implement more sophisticated algorithms
        detections = {}

        # Example: Analyze poses to detect falls
        # This requires pose estimation from YOLOv8 pose model
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            poses = results[0].keypoints.data
            for pose in poses:
                if self.detect_fall(pose.cpu().numpy()):
                    detections["falls"] = {"confidence": 0.8, "time": frame_count / fps}

        # Look for violent actions (simplified)
        boxes = results[0].boxes
        if len(boxes) >= 2:  # Multiple people detected
            if self.detect_attack(boxes):
                detections["attacks"] = {"confidence": 0.7, "time": frame_count / fps}

        return detections

    def detect_fall(self, pose_keypoints):
        # Simple fall detection logic - would be more complex in real application
        # Check if head keypoint is close to ground level
        if len(pose_keypoints) >= 17:  # COCO keypoints format
            head_y = pose_keypoints[0][1]  # Nose keypoint Y coordinate
            ankle_y = max(
                pose_keypoints[15][1], pose_keypoints[16][1]
            )  # Ankle Y coordinates

            # If head is close to feet level and the person is in horizontal position
            if head_y > ankle_y * 0.8:
                return True
        return False

    def detect_attack(self, boxes):
        # Simplified attack detection
        # Would use more sophisticated methods including motion analysis in real app
        if len(boxes) < 2:
            return False

        # Check for close proximity between bounding boxes
        box_centers = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            box_centers.append(center)

        # Check distances between all pairs of boxes
        for i in range(len(box_centers)):
            for j in range(i + 1, len(box_centers)):
                dist = np.sqrt(
                    (box_centers[i][0] - box_centers[j][0]) ** 2
                    + (box_centers[i][1] - box_centers[j][1]) ** 2
                )
                # If people are very close, might be an attack (simplified logic)
                if dist < 100:
                    return True

        return False

    def stop(self):
        self.running = False


class IncidentDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Incident Detection System")
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

        # Top bar with controls
        top_bar = QHBoxLayout()

        # Load video button
        self.load_btn = QPushButton("Load Video")
        self.load_btn.setFixedSize(150, 40)
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
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_video)
        top_bar.addWidget(self.process_btn)

        main_layout.addLayout(top_bar)

        # Main content area with splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Video display area
        video_widget = QWidget()
        video_layout = QVBoxLayout()
        video_widget.setLayout(video_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.video_label.setText("Load a video to begin analysis")
        self.video_label.setMinimumSize(640, 480)

        video_layout.addWidget(self.video_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        video_layout.addWidget(self.progress_bar)

        content_splitter.addWidget(video_widget)

        # Results and visualization tabs
        results_tabs = QTabWidget()

        # Incidents tab
        self.incidents_widget = QWidget()
        incidents_layout = QVBoxLayout()
        self.incidents_widget.setLayout(incidents_layout)

        self.incidents_label = QLabel("Detected incidents will appear here")
        self.incidents_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        incidents_layout.addWidget(self.incidents_label)

        results_tabs.addTab(self.incidents_widget, "Incidents")

        # Visualization tab
        self.viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        self.viz_widget.setLayout(viz_layout)

        # Matplotlib canvas for visualizations
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax.set_title("Incident Timeline")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Incident Type")
        self.ax.grid(True)
        self.canvas.draw()

        viz_layout.addWidget(self.canvas)

        results_tabs.addTab(self.viz_widget, "Visualization")

        content_splitter.addWidget(results_tabs)
        content_splitter.setSizes([600, 400])  # Set initial sizes

        main_layout.addWidget(content_splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

    def load_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if video_path:
            self.video_path = video_path
            self.statusBar().showMessage(f"Loaded: {os.path.basename(video_path)}")

            # Display first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                self.display_frame(frame)
                self.current_frame = frame
            cap.release()

            self.process_btn.setEnabled(True)

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
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)

        self.video_label.setPixmap(scaled_pixmap)

    def draw_detections(self, frame, results):
        # Get the first result (single image)
        result = results[0]

        # Draw bounding boxes and keypoints
        annotated_frame = result.plot()

        return annotated_frame

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

    def process_video(self):
        if self.video_path is None:
            return

        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.process_btn.setText("Process Video")
            return

        # Get confidence threshold
        conf_threshold = self.threshold_slider.value() / 10.0

        # Start processing thread
        self.processing_thread = VideoProcessingThread(
            self.video_path, self.model, conf_threshold
        )

        # Connect signals
        self.processing_thread.update_frame.connect(self.on_frame_processed)
        self.processing_thread.update_progress.connect(self.progress_bar.setValue)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)

        # Update UI
        self.process_btn.setText("Stop Processing")
        self.statusBar().showMessage("Processing video...")

        # Start processing
        self.processing_thread.start()

    def on_frame_processed(self, frame, results):
        self.display_frame(frame, results)

    def on_processing_finished(self, incidents):
        self.incidents = incidents
        self.process_btn.setText("Process Video")
        self.statusBar().showMessage("Processing complete")

        # Update incidents tab
        self.update_incidents_display()

        # Update visualization tab
        self.update_visualization()

    def update_incidents_display(self):
        if not self.incidents:
            return

        # Create text report of incidents
        text = "<h3>Detected Incidents:</h3><br>"

        incident_count = 0
        for incident_type, incidents in self.incidents.items():
            if incidents:
                text += f"<h4>{incident_type.capitalize()}:</h4>"
                text += "<ul>"
                for frame_num, details in incidents:
                    time_sec = frame_num / 30.0  # Assuming 30fps
                    conf = details.get("confidence", 0)
                    text += f"<li>At {time_sec:.2f}s (confidence: {conf:.2f})</li>"
                    incident_count += 1
                text += "</ul>"

        if incident_count == 0:
            text += "<p>No incidents detected.</p>"

        self.incidents_label.setText(text)

    def update_visualization(self):
        # Clear previous plot
        self.ax.clear()

        # Prepare data for timeline visualization
        times = []
        categories = []
        confidences = []

        for incident_type, incidents in self.incidents.items():
            for frame_num, details in incidents:
                time_sec = frame_num / 30.0  # Assuming 30fps
                times.append(time_sec)
                categories.append(incident_type)
                confidences.append(details.get("confidence", 0.5) * 100)

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
            category_numbers = [
                ["falls", "accidents", "attacks"].index(c) for c in categories
            ]

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
            self.ax.set_yticks([0, 1, 2])
            self.ax.set_yticklabels(["Falls", "Accidents", "Attacks"])

            # Set title and labels
            self.ax.set_title("Incident Timeline")
            self.ax.set_xlabel("Time (seconds)")
            self.ax.grid(True)

        # Redraw canvas
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for a cleaner look

    # Set dark theme - using proper PyQt6 palette enums
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
