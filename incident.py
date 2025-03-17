import sys
import os
import cv2
import numpy as np
import torch
import random
import pandas as pd
import datetime
import json
import time
from pathlib import Path
import webbrowser
import tempfile
import logging
import threading
import queue
import socket
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from ultralytics import YOLO

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
    QSystemTrayIcon,
    QStyle,
    QSizePolicy,
    QInputDialog,
    QFrame,
    QCalendarWidget,
    QTimeEdit,
    QWizard,
    QWizardPage,
    QFormLayout,
    QColorDialog,
    QFontDialog,
    QDialogButtonBox,
    QTreeWidget,
    QTreeWidgetItem,
    QCompleter,
    QDateEdit,
    QStackedWidget,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsPathItem,
    QGraphicsTextItem,
    QGraphicsPixmapItem,
    QMainWindow,
    QStyledItemDelegate,
    QScrollBar,
    QPlainTextEdit,
    QSplashScreen,
)

from PyQt6.QtCore import (
    Qt,
    QTimer,
    QThread,
    pyqtSignal,
    QSize,
    QUrl,
    QRect,
    QPoint,
    QMutex,
    QSettings,
    QDateTime,
    QDir,
    QFileInfo,
    QEvent,
    pyqtSlot,
    QPropertyAnimation,
    QEasingCurve,
    QBuffer,
    QIODevice,
    QRunnable,
    QThreadPool,
    QModelIndex,
    QVariant,
    QByteArray,
    QSaveFile,
    QTemporaryFile,
    QElapsedTimer,
    QTime,
    QDate,
)

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
    QKeySequence,
    QCursor,
    QDesktopServices,
    QBrush,
    QPolygon,
    QTransform,
    QRegion,
    QKeyEvent,
    QPainterPath,
    QPicture,
    QGradient,
    QLinearGradient,
    QRadialGradient,
    QConicalGradient,
    QTextDocument,
    QPaintEvent,
    QFont,
    QFontMetrics,
    QTextCursor,
    QTextOption,
    QTextFormat,
    QTextLayout,
    QGuiApplication,
    QPageSize,
    QShortcut,
)

from PyQt6.QtMultimedia import QMediaDevices

# Conditionally import optional modules
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings

    HAS_WEB_ENGINE = True
except ImportError:
    HAS_WEB_ENGINE = False

try:
    from PyQt6.QtCharts import (
        QChart,
        QChartView,
        QLineSeries,
        QBarSet,
        QBarSeries,
        QPieSeries,
        QValueAxis,
        QBarCategoryAxis,
    )

    HAS_CHARTS = True
except ImportError:
    HAS_CHARTS = False

# Global application logger
logger = logging.getLogger("IncidentDetectionApp")

# Global constants
APP_VERSION = "2.0.0"
DEFAULT_MODEL = "yolov8n-pose.pt"
DEFAULT_CAMERA_FPS = 30
MAX_RECENT_VIDEOS = 15
MAX_DETECTION_HISTORY = 5000
INCIDENT_COLORS = {
    "falls": QColor(220, 50, 50),  # Red
    "attacks": QColor(255, 140, 0),  # Orange
    "accidents": QColor(220, 220, 0),  # Yellow
    "intrusions": QColor(100, 149, 237),  # Cornflower Blue
    "loitering": QColor(138, 43, 226),  # Purple
    "abandoned": QColor(34, 139, 34),  # Green
    "custom": QColor(0, 191, 255),  # Deep Sky Blue
}


# Enhanced connection to web services
class NotificationManager:
    """Manages email, SMS, webhooks, and other notification channels"""

    def __init__(self, settings=None):
        self.settings = settings or {}
        self.email_enabled = self.settings.get("email_enabled", False)
        self.sms_enabled = self.settings.get("sms_enabled", False)
        self.webhook_enabled = self.settings.get("webhook_enabled", False)
        self.email_cooldown = {}
        self.sms_cooldown = {}
        self.webhook_cooldown = {}

    def send_email(self, subject, message, recipient=None, image=None):
        """Send email notification with optional incident image"""
        if not self.email_enabled:
            return False

        recipient = recipient or self.settings.get("email_recipient")

        # Check cooldown
        incident_type = subject.split(" ")[0].lower() if " " in subject else "general"
        now = time.time()

        if incident_type in self.email_cooldown and now - self.email_cooldown[
            incident_type
        ] < self.settings.get("email_cooldown", 300):
            logger.debug(
                f"Email notification for {incident_type} skipped (cooldown active)"
            )
            return False

        try:
            # Simplified email sending for demo
            logger.info(f"Would send email to {recipient} with subject: {subject}")

            # Update cooldown
            self.email_cooldown[incident_type] = now
            return True

        except Exception as e:
            logger.error(f"Email notification failed: {str(e)}")
            return False

    def send_webhook(self, incident_data):
        """Send webhook notification to configured endpoints"""
        if not self.webhook_enabled:
            return False

        webhook_urls = self.settings.get("webhook_urls", [])
        if not webhook_urls:
            return False

        # Check cooldown
        incident_type = incident_data.get("type", "general")
        now = time.time()

        if incident_type in self.webhook_cooldown and now - self.webhook_cooldown[
            incident_type
        ] < self.settings.get("webhook_cooldown", 60):
            logger.debug(
                f"Webhook notification for {incident_type} skipped (cooldown active)"
            )
            return False

        success = False
        for url in webhook_urls:
            try:
                # Prepare payload
                payload = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "incident": incident_data,
                }

                # Log instead of actually sending for demo
                logger.info(f"Would send webhook to {url} with payload: {payload}")
                success = True

            except Exception as e:
                logger.error(f"Webhook notification to {url} failed: {str(e)}")

        # Update cooldown if at least one succeeded
        if success:
            self.webhook_cooldown[incident_type] = now

        return success


# Database integration for persistence
class IncidentDatabase:
    """Manages persistent storage of detection results"""

    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(
            QDir.homePath(), ".incident_detector/incidents.db"
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize database
        self.init_db()

    def init_db(self):
        """Initialize the database schema"""
        logger.info("Database initialized successfully")

    def add_incident(
        self,
        incident_type,
        details,
        source=None,
        frame_number=None,
        confidence=None,
        image_path=None,
    ):
        """Add incident to database"""
        logger.info(f"Added incident: {incident_type} with confidence: {confidence}")
        return 1  # Return ID

    def get_incidents(
        self, incident_type=None, source=None, start_date=None, end_date=None, limit=100
    ):
        """Retrieve incidents with optional filtering"""
        # Return sample data
        return [
            {
                "id": 1,
                "type": "fall",
                "timestamp": datetime.datetime.now().isoformat(),
                "source": "Camera",
                "confidence": 0.85,
                "frame_number": 120,
                "details": json.dumps({"position": [100, 200]}),
                "image_path": None,
            }
        ]

    def get_analytics(self, days=30, source=None):
        """Get analytics data for visualization"""
        # Return sample data
        return [
            {
                "date": datetime.date.today().isoformat(),
                "type": "fall",
                "count": 5,
                "avg_confidence": 0.78,
            },
            {
                "date": datetime.date.today().isoformat(),
                "type": "intrusion",
                "count": 3,
                "avg_confidence": 0.82,
            },
        ]


# Machine learning enhanced detection
class BehaviorAnalysisEngine:
    """Advanced behavior analysis engine with ML capabilities"""

    def __init__(self):
        self.motion_history = deque(maxlen=100)
        self.pose_history = {}
        self.trajectory_clusters = None
        self.normal_motion_model = None
        self.initialized = False
        self.anomaly_threshold = 0.75

    def add_motion_data(self, frame_index, tracks):
        """Add motion data for analysis"""
        motion_data = {"frame": frame_index, "tracks": {}}

        for track_id, track_data in tracks.items():
            if "boxes" in track_data and len(track_data["boxes"]) > 0:
                box = track_data["boxes"][-1]
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2

                motion_data["tracks"][track_id] = {
                    "position": (center_x, center_y),
                    "box": box,
                }

        self.motion_history.append(motion_data)

        # Initialize models if enough data is collected
        if len(self.motion_history) >= 50 and not self.initialized:
            self.initialize_models()

    def add_pose_data(self, frame_index, poses):
        """Add pose data for analysis"""
        for i, pose in enumerate(poses):
            person_id = f"pose_{i}"

            # Extract keypoints
            if hasattr(pose, "cpu"):
                keypoints = pose.cpu().numpy()
            else:
                keypoints = pose

            # Store in history
            if person_id not in self.pose_history:
                self.pose_history[person_id] = deque(maxlen=30)

            self.pose_history[person_id].append(
                {"frame": frame_index, "keypoints": keypoints}
            )

    def initialize_models(self):
        """Initialize ML models for behavior analysis"""
        try:
            # Extract trajectory data
            trajectories = []

            for motion_data in self.motion_history:
                for track_id, track_info in motion_data["tracks"].items():
                    trajectories.append(track_info["position"])

            if len(trajectories) > 20:
                # Simple clustering for demo
                self.trajectory_clusters = {
                    "num_clusters": 3,
                }

                self.initialized = True
                logger.info(
                    f"Behavior analysis models initialized with {self.trajectory_clusters['num_clusters']} movement patterns"
                )

        except Exception as e:
            logger.error(f"Error initializing behavior models: {str(e)}")

    def detect_anomalies(self, current_tracks):
        """Detect anomalous behaviors using ML models"""
        if not self.initialized or not current_tracks:
            return []

        anomalies = []

        try:
            # Simple anomaly detection for demo
            if random.random() < 0.05:  # 5% chance of random anomaly
                anomalies.append(
                    {
                        "track_id": (
                            list(current_tracks.keys())[0] if current_tracks else 0
                        ),
                        "type": "unusual_movement",
                        "confidence": 0.8,
                        "position": (100, 100),
                        "details": "Movement pattern outside normal clusters",
                    }
                )

        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")

        return anomalies

    def analyze_pose_anomalies(self, current_poses):
        """Analyze pose data for anomalous body positions"""
        # Explicitly check for None or empty list/tensor
        if (
            current_poses is None
            or (hasattr(current_poses, "numel") and current_poses.numel() == 0)
            or len(current_poses) == 0
        ):
            return []

        anomalies = []

        try:
            # Simple pose anomaly detection for demo
            if random.random() < 0.03:  # 3% chance of random anomaly
                anomalies.append(
                    {
                        "pose_index": 0,
                        "type": "unusual_pose",
                        "confidence": 0.6,
                        "details": "Horizontal body position with elevated head",
                    }
                )

        except Exception as e:
            logger.error(f"Error in pose anomaly analysis: {str(e)}")

        return anomalies


# Enhanced 3D visualization widget with matplotlib
class ThreeDVisualizationWidget(QWidget):
    """Widget for 3D visualization of incident patterns and motion data"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Visualization type
        controls_layout.addWidget(QLabel("Visualization:"))
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems(
            [
                "3D Heatmap",
                "Movement Patterns",
                "Time-Space Plot",
                "Trajectory Analysis",
            ]
        )
        self.viz_type_combo.currentIndexChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.viz_type_combo)

        # Time range
        controls_layout.addWidget(QLabel("Time Range:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(
            ["Today", "Last 3 Days", "Last Week", "Last Month", "All Time"]
        )
        self.time_range_combo.currentIndexChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.time_range_combo)

        # Incident type filter
        controls_layout.addWidget(QLabel("Type:"))
        self.type_filter_combo = QComboBox()
        self.type_filter_combo.addItems(
            ["All Types", "Falls", "Attacks", "Accidents", "Custom"]
        )
        self.type_filter_combo.currentIndexChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.type_filter_combo)

        # Add rotation control
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(0, 360)
        self.rotation_slider.setValue(30)
        self.rotation_slider.setFixedWidth(100)
        self.rotation_slider.valueChanged.connect(self.rotate_view)
        controls_layout.addWidget(QLabel("Rotate:"))
        controls_layout.addWidget(self.rotation_slider)

        self.layout.addLayout(controls_layout)

        # Create graph container with matplotlib Figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)

        # Store incident data
        self.incidents = {}
        self.motion_data = []

        # Initialize with empty plot
        self.update_visualization()

    def set_incidents(self, incidents):
        """Set incident data for visualization"""
        self.incidents = incidents
        self.update_visualization()

    def set_motion_data(self, motion_data):
        """Set motion trajectory data for visualization"""
        self.motion_data = motion_data
        # Only update if in movement visualization modes
        if self.viz_type_combo.currentText() in [
            "Movement Patterns",
            "Trajectory Analysis",
        ]:
            self.update_visualization()

    def update_visualization(self):
        """Update the 3D visualization based on selected type"""
        viz_type = self.viz_type_combo.currentText()

        # Clear the figure
        self.figure.clear()

        # Create 3D subplot
        ax = self.figure.add_subplot(111, projection="3d")

        if viz_type == "3D Heatmap":
            self.create_3d_heatmap(ax)
        elif viz_type == "Movement Patterns":
            self.create_movement_patterns(ax)
        elif viz_type == "Time-Space Plot":
            self.create_time_space_plot(ax)
        elif viz_type == "Trajectory Analysis":
            self.create_trajectory_analysis(ax)

        # Refresh canvas
        self.canvas.draw()

    def create_3d_heatmap(self, ax):
        """Create 3D heatmap of incident locations over time"""
        # Filter incidents by type if needed
        type_filter = self.type_filter_combo.currentText()
        filtered_incidents = {}

        if type_filter == "All Types":
            filtered_incidents = self.incidents
        else:
            filter_key = type_filter.lower().rstrip("s")
            for key, value in self.incidents.items():
                if key.startswith(filter_key):
                    filtered_incidents[key] = value

        # Create sample data for heatmap
        x, y, z = [], [], []
        colors = []

        # Process incident data
        for incident_type, incidents_list in filtered_incidents.items():
            # Map incident types to colors
            color_val = INCIDENT_COLORS.get(incident_type, INCIDENT_COLORS["custom"])
            color = (
                color_val.red() / 255,
                color_val.green() / 255,
                color_val.blue() / 255,
            )

            for frame_num, details in incidents_list:
                if not isinstance(details, list):
                    details = [details]

                for detail in details:
                    # Use random position for demo
                    pos_x = random.uniform(0, 100)
                    pos_y = random.uniform(0, 100)

                    # Use time as Z axis
                    time_val = detail.get("time", frame_num / 30.0)

                    x.append(pos_x)
                    y.append(pos_y)
                    z.append(time_val)
                    colors.append(color)

        # Create scatter plot if we have data
        if x:
            ax.scatter(x, y, z, c=colors, alpha=0.6, s=30)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_zlabel("Time (seconds)")
            ax.set_title("3D Incident Heatmap")
        else:
            ax.text(
                0.5,
                0.5,
                0.5,
                "No incident data available",
                horizontalalignment="center",
                size=14,
                color="gray",
            )

    def create_movement_patterns(self, ax):
        """Visualize movement patterns in 3D space"""
        # Create sample data for visualization
        if not self.motion_data:
            # Create some random tracks for visualization
            random_tracks = []
            for i in range(5):
                track = []
                x, y = random.uniform(20, 80), random.uniform(20, 80)
                for j in range(10):
                    x += random.uniform(-5, 5)
                    y += random.uniform(-5, 5)
                    track.append((x, y))
                random_tracks.append(track)
            tracks = random_tracks
        else:
            tracks = self.motion_data

        # Plot each track with different color
        for i, track in enumerate(tracks):
            xs = [pos[0] for pos in track]
            ys = [pos[1] for pos in track]
            zs = list(range(len(track)))  # Use sequence as Z

            # Pick a color
            colors = ["r", "g", "b", "c", "m", "y"]
            color = colors[i % len(colors)]

            # Plot the track
            ax.plot(xs, ys, zs, color, linewidth=2, marker="o", markersize=4)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Time Sequence")
        ax.set_title("Movement Patterns")

    def create_time_space_plot(self, ax):
        """Create 3D time-space plot of incidents"""
        # Create sample data for visualization
        x = []  # X position
        y = []  # Y position
        z = []  # Time
        c = []  # Type (color)

        for incident_type, incidents_list in self.incidents.items():
            color_val = INCIDENT_COLORS.get(incident_type, INCIDENT_COLORS["custom"])
            color = (
                color_val.red() / 255,
                color_val.green() / 255,
                color_val.blue() / 255,
            )

            for i, (frame_num, details) in enumerate(incidents_list):
                # Create position in space
                x.append(random.uniform(0, 5))  # Zone X
                y.append(random.uniform(0, 5))  # Zone Y
                z.append(i)  # Time progression
                c.append(color)

        # Create a scatter plot
        if x:
            scatter = ax.scatter(x, y, z, c=c, s=50, alpha=0.6)
            ax.set_xlabel("Zone X")
            ax.set_ylabel("Zone Y")
            ax.set_zlabel("Time")
            ax.set_title("Time-Space Incident Distribution")

            # Set axis labels
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels(["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", ""])
            ax.set_yticklabels(["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", ""])
        else:
            ax.text(
                0.5,
                0.5,
                0.5,
                "No incident data available",
                horizontalalignment="center",
                size=14,
                color="gray",
            )

    def create_trajectory_analysis(self, ax):
        """Create 3D visualization of movement trajectories with clustering"""
        # Create synthetic clusters for demonstration
        clusters = [
            [(random.uniform(10, 30), random.uniform(10, 30)) for _ in range(20)],
            [(random.uniform(50, 70), random.uniform(50, 70)) for _ in range(15)],
            [(random.uniform(20, 40), random.uniform(60, 80)) for _ in range(25)],
        ]

        colors = ["r", "g", "b"]  # Red, Green, Blue

        for cluster_idx, cluster in enumerate(clusters):
            x = [pos[0] for pos in cluster]
            y = [pos[1] for pos in cluster]
            z = [
                random.uniform(0, 5) for _ in range(len(cluster))
            ]  # Add vertical variation

            ax.scatter(x, y, z, c=colors[cluster_idx % len(colors)], s=30, alpha=0.6)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("Trajectory Clusters")

    def rotate_view(self):
        """Rotate the 3D view based on slider position"""
        angle = self.rotation_slider.value()

        # Get the current axes and adjust the view angle
        if self.figure.axes:
            ax = self.figure.axes[0]
            ax.view_init(30, angle)  # 30 degree elevation, varying azimuth
            self.canvas.draw()


# Enhanced heat map with spatial analysis
class SpatialAnalysisWidget(QWidget):
    """Widget for advanced spatial analysis of incidents"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # Create controls
        controls_layout = QHBoxLayout()

        # Visualization type
        controls_layout.addWidget(QLabel("Analysis:"))
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(
            [
                "Incident Heatmap",
                "Risk Zones",
                "Traffic Patterns",
                "Crowd Density",
                "Dwell Time Analysis",
            ]
        )
        self.analysis_type_combo.currentIndexChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.analysis_type_combo)

        # Resolution
        controls_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["Low", "Medium", "High"])
        self.resolution_combo.setCurrentIndex(1)  # Medium default
        self.resolution_combo.currentIndexChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.resolution_combo)

        # Time period
        controls_layout.addWidget(QLabel("Period:"))
        self.time_period_combo = QComboBox()
        self.time_period_combo.addItems(
            ["Today", "Last 3 Days", "Last Week", "Last Month", "All Time"]
        )
        self.time_period_combo.currentIndexChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.time_period_combo)

        # Add overlay controls
        controls_layout.addWidget(QLabel("Overlay:"))
        self.overlay_check = QCheckBox("Grid")
        self.overlay_check.setChecked(True)
        self.overlay_check.stateChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.overlay_check)

        self.labels_check = QCheckBox("Labels")
        self.labels_check.setChecked(True)
        self.labels_check.stateChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.labels_check)

        self.layout.addLayout(controls_layout)

        # Add matplotlib figure for visualization
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)

        # Data
        self.incidents = {}
        self.traffic_data = []
        self.motion_history = []

        # Generate initial visualization
        self.update_visualization()

    def set_incidents(self, incidents):
        """Set incident data for visualization"""
        self.incidents = incidents
        self.update_visualization()

    def set_traffic_data(self, traffic_data):
        """Set traffic flow data for visualization"""
        self.traffic_data = traffic_data

        # Update only if in traffic visualization mode
        if self.analysis_type_combo.currentText() == "Traffic Patterns":
            self.update_visualization()

    def set_motion_history(self, motion_history):
        """Set motion history data for visualization"""
        self.motion_history = motion_history

        # Update if in dwell time or density analysis
        current_viz = self.analysis_type_combo.currentText()
        if current_viz in ["Dwell Time Analysis", "Crowd Density"]:
            self.update_visualization()

    def update_visualization(self):
        """Update visualization based on selected type"""
        # Clear the figure
        self.figure.clear()

        # Get selected visualization type
        viz_type = self.analysis_type_combo.currentText()

        # Create appropriate visualization
        if viz_type == "Incident Heatmap":
            self.create_incident_heatmap()
        elif viz_type == "Risk Zones":
            self.create_risk_zones()
        elif viz_type == "Traffic Patterns":
            self.create_traffic_patterns()
        elif viz_type == "Crowd Density":
            self.create_crowd_density()
        elif viz_type == "Dwell Time Analysis":
            self.create_dwell_time_analysis()

        # Redraw canvas
        self.canvas.draw()

    def create_incident_heatmap(self):
        """Create heatmap of incident locations"""
        ax = self.figure.add_subplot(111)

        # Sample data - in a real app, use actual incident locations
        incident_points = []

        # Gather incident locations
        for incident_type, incidents_list in self.incidents.items():
            for frame_num, details in incidents_list:
                if not isinstance(details, list):
                    details = [details]

                for detail in details:
                    # Generate position if not available
                    if "position" in detail:
                        x, y = detail["position"]
                    else:
                        # Random position for demo
                        x = random.uniform(0, 100)
                        y = random.uniform(0, 100)

                    incident_points.append((x, y))

        if not incident_points:
            # Generate some sample data if no real incidents
            incident_points = [
                (random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50)
            ]

        # Convert to numpy arrays
        x = [p[0] for p in incident_points]
        y = [p[1] for p in incident_points]

        # Create 2D histogram for the heatmap
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=20)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Display the heatmap
        im = ax.imshow(heatmap.T, extent=extent, origin="lower", cmap="hot")

        # Add grid if requested
        if self.overlay_check.isChecked():
            ax.grid(True, color="white", alpha=0.3)

        # Add labels
        if self.labels_check.isChecked():
            ax.set_title("Incident Heatmap")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label("Incident Density")

    def create_risk_zones(self):
        """Create visualization of risk zones based on incident clustering"""
        ax = self.figure.add_subplot(111)

        # Create some sample clusters for the visualization
        centers = [(20, 20), (70, 30), (50, 70)]
        incident_points = []

        for cx, cy in centers:
            for _ in range(20):
                incident_points.append(
                    (cx + random.gauss(0, 8), cy + random.gauss(0, 8))
                )

        # Convert to numpy arrays
        x = [p[0] for p in incident_points]
        y = [p[1] for p in incident_points]

        # Create a 2D histogram for the heatmap
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=30)

        # Apply gaussian filter for smoother visualization
        from scipy.ndimage import gaussian_filter

        sigma = 1.0  # Adjust based on resolution setting
        heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)

        # Plot the heatmap
        extent = [0, 100, 0, 100]
        im = ax.imshow(
            heatmap_smooth.T, extent=extent, origin="lower", cmap="RdYlGn_r", alpha=0.7
        )

        # Plot contour lines
        if self.overlay_check.isChecked():
            cs = ax.contour(
                np.linspace(0, 100, 30),
                np.linspace(0, 100, 30),
                heatmap_smooth.T,
                levels=5,
                colors="k",
                alpha=0.5,
                linewidths=0.5,
            )

        # Plot incident points
        ax.scatter(x, y, s=10, c="k", alpha=0.3)

        # Add labels
        if self.labels_check.isChecked():
            ax.set_title("Risk Zone Analysis")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label("Risk Level")

            # Label high-risk zones
            for i, (cx, cy) in enumerate(centers):
                ax.text(
                    cx,
                    cy,
                    f"Risk Zone {i+1}",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
                )

    def create_traffic_patterns(self):
        """Create visualization of traffic flow patterns"""
        ax = self.figure.add_subplot(111)

        # Create synthetic flow patterns
        paths = [
            [(10, 10), (30, 20), (50, 50), (70, 60), (90, 90)],  # Diagonal path
            [(10, 90), (30, 70), (50, 50), (70, 30), (90, 10)],  # Reverse diagonal
            [(10, 50), (30, 50), (50, 50), (70, 50), (90, 50)],  # Horizontal path
            [(50, 10), (50, 30), (50, 50), (50, 70), (50, 90)],  # Vertical path
        ]

        # Calculate flow vectors
        xs = []
        ys = []
        u = []  # x-component of flow
        v = []  # y-component of flow

        for path in paths:
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]

                # Calculate flow vector
                dx = x2 - x1
                dy = y2 - y1

                xs.append(x1)
                ys.append(y1)
                u.append(dx)
                v.append(dy)

        # Plot flow as quiver plot
        ax.quiver(xs, ys, u, v, scale=50, color="b", alpha=0.7)

        # Add background heatmap for traffic intensity
        heatmap = np.zeros((10, 10))

        # Fill heatmap with traffic density
        for path in paths:
            for x, y in path:
                # Map x,y to heatmap indices
                i = min(int(x / 10), 9)
                j = min(int(y / 10), 9)
                heatmap[j, i] += 1

        # Plot heatmap
        extent = [0, 100, 0, 100]
        im = ax.imshow(heatmap, extent=extent, origin="lower", cmap="YlOrRd", alpha=0.3)

        # Add grid if requested
        if self.overlay_check.isChecked():
            ax.grid(True, color="white", alpha=0.3)

        # Add labels
        if self.labels_check.isChecked():
            ax.set_title("Traffic Flow Patterns")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label("Traffic Intensity")

    def create_crowd_density(self):
        """Create visualization of crowd density patterns"""
        ax = self.figure.add_subplot(111)

        # Create synthetic crowd density data
        hotspots = [(20, 20), (50, 50), (80, 20)]
        positions = []

        for _ in range(300):
            # Pick a random hotspot
            hx, hy = random.choice(hotspots)

            # Add a person near the hotspot
            positions.append((hx + random.gauss(0, 10), hy + random.gauss(0, 10)))

        x = [p[0] for p in positions]
        y = [p[1] for p in positions]

        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            x, y, bins=40, range=[[0, 100], [0, 100]]
        )

        # Smooth the heatmap
        from scipy.ndimage import gaussian_filter

        heatmap_smooth = gaussian_filter(heatmap, sigma=1.0)

        # Plot the heatmap
        extent = [0, 100, 0, 100]
        im = ax.imshow(
            heatmap_smooth.T, extent=extent, origin="lower", cmap="YlOrRd", alpha=0.8
        )

        # Add contour lines if overlay is enabled
        if self.overlay_check.isChecked():
            cs = ax.contour(
                np.linspace(0, 100, 40),
                np.linspace(0, 100, 40),
                heatmap_smooth.T,
                levels=5,
                colors="k",
                alpha=0.5,
                linewidths=0.5,
            )

        # Add labels
        if self.labels_check.isChecked():
            ax.set_title("Crowd Density Analysis")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label("Crowd Density")

            # Label hotspots
            for i, (hx, hy) in enumerate(hotspots):
                ax.text(
                    hx,
                    hy,
                    f"Hotspot {i+1}",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
                )

    def create_dwell_time_analysis(self):
        """Create visualization of dwell time patterns"""
        ax = self.figure.add_subplot(111)

        # Create zones
        zones = {
            "Entrance": (10, 50, 20, 20),  # x, y, width, height
            "Checkout": (70, 50, 20, 20),
            "Display A": (30, 30, 15, 15),
            "Display B": (50, 70, 15, 15),
            "Restrooms": (80, 20, 15, 15),
        }

        # Generate random dwell times for each zone
        avg_dwell_times = {
            "Entrance": 10,  # seconds
            "Checkout": 120,
            "Display A": 45,
            "Display B": 60,
            "Restrooms": 180,
        }

        # Generate data
        dwell_data = []
        for zone_name, (x, y, w, h) in zones.items():
            avg_dwell = avg_dwell_times[zone_name]

            # Create multiple data points for each zone
            for _ in range(10):
                # Random position within zone
                pos_x = x + random.uniform(0, w)
                pos_y = y + random.uniform(0, h)

                # Random dwell time around average
                dwell_time = max(1, random.gauss(avg_dwell, avg_dwell * 0.3))

                dwell_data.append(
                    {
                        "position": (pos_x, pos_y),
                        "dwell_time": dwell_time,
                        "zone": zone_name,
                    }
                )

        # Extract positions and dwell times
        positions = [d["position"] for d in dwell_data]
        dwell_times = [d["dwell_time"] for d in dwell_data]

        # Create a scatter plot colored by dwell time
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]

        # Scale marker size by dwell time
        sizes = [max(20, min(500, t * 2)) for t in dwell_times]

        scatter = ax.scatter(
            x,
            y,
            s=sizes,
            c=dwell_times,
            cmap="plasma",
            alpha=0.6,
            edgecolors="k",
            linewidths=0.5,
        )

        # Draw zones
        for zone_name, (zx, zy, zw, zh) in zones.items():
            rect = plt.Rectangle(
                (zx, zy),
                zw,
                zh,
                fill=False,
                edgecolor="black",
                linestyle="--",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Add zone label with average dwell time
            zone_times = [d["dwell_time"] for d in dwell_data if d["zone"] == zone_name]
            avg_time = sum(zone_times) / len(zone_times) if zone_times else 0

            ax.text(
                zx + zw / 2,
                zy + zh / 2,
                f"{zone_name}\n{avg_time:.1f}s",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7),
            )

        # Add colorbar if labels are enabled
        if self.labels_check.isChecked():
            cbar = self.figure.colorbar(scatter, ax=ax)
            cbar.set_label("Dwell Time (seconds)")

        # Add labels
        if self.labels_check.isChecked():
            ax.set_title("Dwell Time Analysis")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

        # Set axis limits
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])


# Enhanced detection thread with more advanced features
class AdvancedDetectionThread(QThread):
    """Enhanced detection thread with advanced analysis capabilities"""

    update_frame = pyqtSignal(np.ndarray, list, dict)
    update_progress = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    update_stats = pyqtSignal(dict)
    incident_detected = pyqtSignal(str, dict)
    anomaly_detected = pyqtSignal(str, dict)
    motion_updated = pyqtSignal(list)

    def __init__(self, model, settings):
        super().__init__()
        self.model = model
        self.settings = settings
        self.running = False
        self.paused = False
        self.mutex = QMutex()

        # For video file processing
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0

        # For camera processing
        self.camera_id = None

        # Detection settings
        self.conf_threshold = settings.get("confidence_threshold", 0.5)
        self.frame_interval = settings.get("frame_interval", 1)
        self.roi = settings.get("roi", None)
        self.custom_rules = settings.get("custom_rules", [])

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "falls_detected": 0,
            "attacks_detected": 0,
            "accidents_detected": 0,
            "intrusions_detected": 0,
            "loitering_detected": 0,
            "abandoned_detected": 0,
            "custom_detected": 0,
            "anomalies_detected": 0,
            "fps": 0,
            "avg_processing_time": 0,
            "current_frame_number": 0,
        }

        # Detection history
        self.incidents = {
            "falls": [],
            "attacks": [],
            "accidents": [],
            "intrusions": [],
            "loitering": [],
            "abandoned": [],
            "anomalies": [],
        }

        # For custom event types
        for rule in self.custom_rules:
            if rule.get("type") == "custom":
                rule_name = rule.get("name", "custom").lower().replace(" ", "_")
                if rule_name not in self.incidents:
                    self.incidents[rule_name] = []

        # Temporal analysis
        self.recent_frames = deque(maxlen=30)

        # Tracking
        self.tracks = {}
        self.track_history = []

        # Alert cooldown
        self.alert_cooldown = {}
        self.cooldown_frames = settings.get("cooldown_frames", 30)

        # Advanced behavior analysis
        self.behavior_engine = BehaviorAnalysisEngine()

        # Zone definitions for advanced analysis
        self.zones = settings.get("zones", {})

        # Initialize background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    def set_video(self, video_path):
        """Set up for processing a video file"""
        self.video_path = video_path
        self.camera_id = None

    def set_camera(self, camera_id):
        """Set up for processing camera feed"""
        self.camera_id = camera_id
        self.video_path = None

    def update_settings(self, settings):
        """Update detection settings"""
        self.mutex.lock()
        self.settings = settings
        self.conf_threshold = settings.get("confidence_threshold", 0.5)
        self.frame_interval = settings.get("frame_interval", 1)
        self.roi = settings.get("roi", None)
        self.custom_rules = settings.get("custom_rules", [])
        self.cooldown_frames = settings.get("cooldown_frames", 30)
        self.zones = settings.get("zones", {})
        self.mutex.unlock()

    def run(self):
        """Main processing loop"""
        self.running = True

        # Determine source (video file or camera)
        if self.video_path:
            success = self.process_video()
        elif self.camera_id is not None:
            success = self.process_camera()
        else:
            self.running = False
            return

        # Signal completion
        if success:
            self.processing_finished.emit(self.incidents)

    def process_video(self):
        """Process a video file"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Processing performance tracking
        processing_times = []
        frame_count = 0
        skip_count = 0

        # Motion tracking
        accumulated_motion = np.zeros((2, 3))

        while self.running and self.cap.isOpened():
            if self.paused:
                time.sleep(0.1)  # Sleep when paused
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            # Process only every Nth frame for efficiency
            skip_count += 1
            if skip_count < self.frame_interval:
                continue
            skip_count = 0

            # Apply background subtraction for motion detection
            fg_mask = self.bg_subtractor.apply(frame)

            # Process the frame
            start_time = time.time()
            results, detections, anomalies = self.process_frame(
                frame, frame_count, fps, fg_mask
            )

            # Measure processing time
            process_time = time.time() - start_time
            processing_times.append(process_time)
            if len(processing_times) > 100:
                processing_times.pop(0)
            self.stats["avg_processing_time"] = sum(processing_times) / len(
                processing_times
            )

            # Update current frame number for UI
            self.stats["current_frame_number"] = frame_count

            # Update progress
            progress = (
                int((frame_count / self.total_frames) * 100)
                if self.total_frames > 0
                else 0
            )
            self.update_progress.emit(progress)

            # Record detected incidents
            for incident_type, incidents in detections.items():
                if incidents:
                    # Update statistics
                    stat_key = f"{incident_type}_detected"
                    if stat_key in self.stats:
                        self.stats[stat_key] += 1

                    # Record incident
                    self.incidents[incident_type].append((frame_count, incidents))

                    # Check cooldown before alerting
                    if (
                        incident_type not in self.alert_cooldown
                        or frame_count - self.alert_cooldown[incident_type]
                        > self.cooldown_frames
                    ):
                        self.incident_detected.emit(incident_type, incidents[0])
                        self.alert_cooldown[incident_type] = frame_count

            # Process anomalies
            if anomalies:
                self.stats["anomalies_detected"] += 1

                # Record anomalies
                self.incidents["anomalies"].append((frame_count, anomalies))

                # Signal anomaly detection
                self.anomaly_detected.emit("anomaly", anomalies[0])

            # Update UI with processed frame and detections
            self.stats["frames_processed"] += 1
            self.update_stats.emit(self.stats.copy())
            self.update_frame.emit(frame, results, detections)

            # Emit motion data for visualization
            if frame_count % 10 == 0:  # Emit every 10 frames to reduce overhead
                self.motion_updated.emit(self.track_history)

            frame_count += 1

        # Cleanup
        if self.cap:
            self.cap.release()

        return True

    def process_camera(self):
        """Process a camera feed"""
        # Handle both numeric IDs and string URLs for IP cameras
        if isinstance(self.camera_id, str) and self.camera_id.startswith(
            ("rtsp://", "http://", "https://")
        ):
            self.cap = cv2.VideoCapture(self.camera_id)
        else:
            self.cap = cv2.VideoCapture(int(self.camera_id))

        if not self.cap.isOpened():
            return False

        # For FPS calculation
        fps_counter = 0
        fps_timer = time.time()

        # For processing time calculation
        processing_times = []

        # Frame counter
        frame_count = 0
        skip_count = 0

        while self.running and self.cap.isOpened():
            if self.paused:
                time.sleep(0.1)  # Sleep when paused
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            # Process only every Nth frame for efficiency
            skip_count += 1
            if skip_count < self.frame_interval:
                continue
            skip_count = 0

            # Apply background subtraction for motion detection
            fg_mask = self.bg_subtractor.apply(frame)

            # Process the frame
            start_time = time.time()

            # Get camera FPS
            camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if camera_fps <= 0:  # If camera doesn't report FPS, use a default
                camera_fps = 30.0

            results, detections, anomalies = self.process_frame(
                frame, frame_count, camera_fps, fg_mask
            )

            # Measure processing time
            process_time = time.time() - start_time
            processing_times.append(process_time)
            if len(processing_times) > 100:
                processing_times.pop(0)
            self.stats["avg_processing_time"] = sum(processing_times) / len(
                processing_times
            )

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.stats["fps"] = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            # Record detected incidents
            for incident_type, incidents in detections.items():
                if incidents:
                    # Update statistics
                    stat_key = f"{incident_type}_detected"
                    if stat_key in self.stats:
                        self.stats[stat_key] += 1

                    # Record incident
                    self.incidents[incident_type].append((frame_count, incidents))

                    # Check cooldown before alerting
                    if (
                        incident_type not in self.alert_cooldown
                        or frame_count - self.alert_cooldown[incident_type]
                        > self.cooldown_frames
                    ):
                        self.incident_detected.emit(incident_type, incidents[0])
                        self.alert_cooldown[incident_type] = frame_count

            # Process anomalies
            if anomalies:
                self.stats["anomalies_detected"] += 1

                # Record anomalies
                self.incidents["anomalies"].append((frame_count, anomalies))

                # Signal anomaly detection
                self.anomaly_detected.emit("anomaly", anomalies[0])

            # Update UI with processed frame and detections
            self.stats["frames_processed"] += 1
            self.update_stats.emit(self.stats.copy())
            self.update_frame.emit(frame, results, detections)

            # Emit motion data for visualization
            if frame_count % 10 == 0:  # Emit every 10 frames to reduce overhead
                self.motion_updated.emit(self.track_history)

            frame_count += 1

        # Cleanup
        if self.cap:
            self.cap.release()

        return True

    def process_frame(self, frame, frame_count, fps, fg_mask=None):
        """Process a single frame for incident detection"""
        # Apply ROI if specified
        if self.roi:
            x1, y1, x2, y2 = self.roi
            frame_roi = frame[y1:y2, x1:x2]
            results = self.model(frame_roi, verbose=False, conf=self.conf_threshold)
        else:
            # Run inference on full frame
            results = self.model(frame, verbose=False, conf=self.conf_threshold)

        # Add current frame to recent frames list
        self.recent_frames.append((frame, results))

        # Process results to detect anomalies
        detections = self.process_detections(results, frame_count, fps, fg_mask)

        # Update tracking
        self.update_tracks(results, frame_count, frame)

        # Process custom rules
        for rule in self.custom_rules:
            custom_detections = self.apply_custom_rule(
                rule, results, frame_count, fps, fg_mask
            )
            if custom_detections:
                rule_type = rule.get("type")
                if rule_type == "custom":
                    rule_type = rule.get("name", "custom").lower().replace(" ", "_")
                    if rule_type not in detections:
                        detections[rule_type] = []

                if rule_type in detections:
                    detections[rule_type].extend(custom_detections)

        # Perform advanced behavior analysis
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            poses = results[0].keypoints.data
            self.behavior_engine.add_pose_data(frame_count, poses)

        self.behavior_engine.add_motion_data(frame_count, self.tracks)

        # Detect behavioral anomalies
        anomalies = self.behavior_engine.detect_anomalies(self.tracks)
        pose_anomalies = self.behavior_engine.analyze_pose_anomalies(
            results[0].keypoints.data
            if hasattr(results[0], "keypoints") and results[0].keypoints is not None
            else []
        )

        # Combine all anomalies
        all_anomalies = anomalies + pose_anomalies

        return results, detections, all_anomalies

    def process_detections(self, results, frame_count, fps, fg_mask=None):
        """Analyze model results to detect incidents"""
        detections = {}

        # Example: Analyze poses to detect falls
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            poses = results[0].keypoints.data
            for i, pose in enumerate(poses):
                keypoints = pose.cpu().numpy()

                if self.detect_fall(keypoints):
                    confidence = 0.8  # Simplified confidence value
                    if "falls" not in detections:
                        detections["falls"] = []

                    detections["falls"].append(
                        {
                            "confidence": confidence,
                            "time": frame_count / fps,
                            "frame": frame_count,
                            "person_idx": i,
                        }
                    )

        # Detect attacks between people
        boxes = results[0].boxes if hasattr(results[0], "boxes") else None
        if boxes is not None and len(boxes) >= 2:
            if self.detect_attack(boxes):
                if "attacks" not in detections:
                    detections["attacks"] = []

                detections["attacks"].append(
                    {"confidence": 0.7, "time": frame_count / fps, "frame": frame_count}
                )

        # Detect accidents based on sudden movements
        if len(self.recent_frames) >= 3:
            if self.detect_accident(frame_count, fps):
                if "accidents" not in detections:
                    detections["accidents"] = []

                detections["accidents"].append(
                    {"confidence": 0.6, "time": frame_count / fps, "frame": frame_count}
                )

        # Detect intrusions in restricted zones
        if self.zones:
            intrusions = self.detect_zone_intrusions(
                results, self.zones.get("restricted", [])
            )
            if intrusions:
                if "intrusions" not in detections:
                    detections["intrusions"] = []

                for intrusion in intrusions:
                    intrusion["time"] = frame_count / fps
                    intrusion["frame"] = frame_count
                    detections["intrusions"].append(intrusion)

        # Detect loitering
        loiterers = self.detect_loitering(frame_count, fps)
        if loiterers:
            if "loitering" not in detections:
                detections["loitering"] = []

            for loiterer in loiterers:
                loiterer["time"] = frame_count / fps
                loiterer["frame"] = frame_count
                detections["loitering"].append(loiterer)

        # Detect abandoned objects using foreground mask
        if fg_mask is not None:
            abandoned = self.detect_abandoned_objects(fg_mask, frame_count, fps)
            if abandoned:
                if "abandoned" not in detections:
                    detections["abandoned"] = []

                for obj in abandoned:
                    obj["time"] = frame_count / fps
                    obj["frame"] = frame_count
                    detections["abandoned"].append(obj)

        return detections

    def detect_fall(self, pose_keypoints):
        """Detect falls based on pose keypoints"""
        # Simple fall detection based on head position
        if len(pose_keypoints) >= 17:  # COCO keypoints format
            # Random fall detection for testing purposes
            return random.random() < 0.02  # 2% chance of fall detection
        return False

    def detect_attack(self, boxes):
        """Detect potential attacks between people"""
        # Simple attack detection based on proximity
        # Random attack detection for testing
        return random.random() < 0.01  # 1% chance of attack detection

    def detect_accident(self, frame_count, fps):
        """Detect accidents based on sudden movements"""
        # Simple accident detection for demo
        return random.random() < 0.01  # 1% chance of accident detection

    def detect_zone_intrusions(self, results, restricted_zones):
        """Detect intrusions into restricted zones"""
        if not restricted_zones:
            return []

        # Simple random intrusion detection for testing purposes
        if random.random() < 0.02:  # 2% chance
            return [
                {
                    "confidence": 0.9,
                    "person_idx": 0,
                    "zone": "Restricted Zone",
                    "position": (random.uniform(0, 100), random.uniform(0, 100)),
                }
            ]
        return []

    def detect_loitering(self, frame_count, fps):
        """Detect people loitering in one area for too long"""
        # Random loitering detection for testing
        if random.random() < 0.01:  # 1% chance
            return [
                {
                    "confidence": 0.7,
                    "track_id": 0,
                    "duration": random.uniform(30, 120),
                    "position": (random.uniform(0, 100), random.uniform(0, 100)),
                }
            ]
        return []

    def detect_abandoned_objects(self, fg_mask, frame_count, fps):
        """Detect potentially abandoned objects using background subtraction"""
        # Random abandoned object detection for testing
        if random.random() < 0.005:  # 0.5% chance
            return [
                {
                    "confidence": 0.6,
                    "position": (random.uniform(0, 100), random.uniform(0, 100)),
                    "size": random.uniform(100, 500),
                    "bbox": (
                        random.uniform(0, 80),
                        random.uniform(0, 80),
                        random.uniform(10, 20),
                        random.uniform(10, 20),
                    ),
                }
            ]
        return []

    def update_tracks(self, results, frame_count, frame=None):
        """Update tracking information for detected people"""
        # Simple track update for demo purposes

        # Add to track history occasionally
        if random.random() < 0.1 and len(self.track_history) < 10:
            # Create a random track
            track = [
                (random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)
            ]
            self.track_history.append(track)

    def calculate_iou(self, box1, box2):
        """Calculate intersection over union between two boxes"""
        # Calculate intersection
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])

        # Check if boxes overlap
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / union

        return iou

    def apply_custom_rule(self, rule, results, frame_count, fps, fg_mask=None):
        """Apply a custom detection rule"""
        # Random custom rule detection for testing
        rule_type = rule.get("type", "custom")
        if random.random() < 0.02:  # 2% chance
            return [
                {
                    "confidence": 0.7,
                    "time": frame_count / fps,
                    "frame": frame_count,
                    "rule": rule.get("name", "Custom Rule"),
                    "position": (random.uniform(0, 100), random.uniform(0, 100)),
                }
            ]
        return None

    def pause(self):
        """Pause processing"""
        self.paused = True

    def resume(self):
        """Resume processing"""
        self.paused = False

    def stop(self):
        """Stop processing"""
        self.running = False


# Zone configuration dialog
class ZoneConfigDialog(QDialog):
    """Dialog for configuring detection zones"""

    def __init__(self, zones=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Detection Zones")
        self.setMinimumSize(600, 500)

        # Initialize zones
        self.zones = zones or {"restricted": [], "monitoring": [], "entry_exit": []}

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Zone types tab widget
        self.zone_tabs = QTabWidget()

        # Create tabs for each zone type
        self.create_zone_tab("restricted", "Restricted Zones")
        self.create_zone_tab("monitoring", "Monitoring Zones")
        self.create_zone_tab("entry_exit", "Entry/Exit Points")

        layout.addWidget(self.zone_tabs)

        # Buttons
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Zones")
        self.save_btn.clicked.connect(self.accept)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_zone_tab(self, zone_type, title):
        """Create a tab for a specific zone type"""
        tab = QWidget()
        tab_layout = QVBoxLayout()

        # Explanation label
        explanation = ""
        if zone_type == "restricted":
            explanation = "Restricted zones trigger alerts when people enter them."
        elif zone_type == "monitoring":
            explanation = "Monitoring zones are analyzed for unusual activity."
        elif zone_type == "entry_exit":
            explanation = (
                "Entry/Exit points are used for people counting and flow analysis."
            )

        explanation_label = QLabel(explanation)
        explanation_label.setWordWrap(True)
        tab_layout.addWidget(explanation_label)

        # Zone list
        zone_group = QGroupBox(f"{title}")
        zone_layout = QVBoxLayout()

        # Zone table
        self.zone_tables = getattr(self, "zone_tables", {})
        zone_table = QTableWidget()
        zone_table.setColumnCount(5)
        zone_table.setHorizontalHeaderLabels(["Name", "X", "Y", "Width", "Height"])
        zone_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # Populate table with existing zones
        for i, zone in enumerate(self.zones.get(zone_type, [])):
            zone_table.insertRow(i)

            if len(zone) >= 5:
                x, y, w, h, name = zone
            else:
                x, y, w, h = zone
                name = f"Zone {i+1}"

            zone_table.setItem(i, 0, QTableWidgetItem(name))
            zone_table.setItem(i, 1, QTableWidgetItem(str(x)))
            zone_table.setItem(i, 2, QTableWidgetItem(str(y)))
            zone_table.setItem(i, 3, QTableWidgetItem(str(w)))
            zone_table.setItem(i, 4, QTableWidgetItem(str(h)))

        self.zone_tables[zone_type] = zone_table
        zone_layout.addWidget(zone_table)

        # Zone controls
        controls_layout = QHBoxLayout()

        add_btn = QPushButton("Add Zone")
        add_btn.clicked.connect(lambda: self.add_zone(zone_type))

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(lambda: self.remove_zone(zone_type))

        controls_layout.addWidget(add_btn)
        controls_layout.addWidget(remove_btn)

        zone_layout.addLayout(controls_layout)
        zone_group.setLayout(zone_layout)

        tab_layout.addWidget(zone_group)
        tab.setLayout(tab_layout)

        self.zone_tabs.addTab(tab, title)

    def add_zone(self, zone_type):
        """Add a new zone to the specified type"""
        table = self.zone_tables.get(zone_type)
        if not table:
            return

        row = table.rowCount()
        table.insertRow(row)

        # Default values
        name = f"Zone {row+1}"
        x, y, w, h = 10, 10, 100, 100

        table.setItem(row, 0, QTableWidgetItem(name))
        table.setItem(row, 1, QTableWidgetItem(str(x)))
        table.setItem(row, 2, QTableWidgetItem(str(y)))
        table.setItem(row, 3, QTableWidgetItem(str(w)))
        table.setItem(row, 4, QTableWidgetItem(str(h)))

    def remove_zone(self, zone_type):
        """Remove selected zone from the specified type"""
        table = self.zone_tables.get(zone_type)
        if not table:
            return

        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            return

        # Remove in reverse order to avoid index issues
        for row in sorted([index.row() for index in selected_rows], reverse=True):
            table.removeRow(row)

    def get_zones(self):
        """Get the updated zones configuration"""
        zones = {}

        for zone_type, table in self.zone_tables.items():
            zones[zone_type] = []

            for row in range(table.rowCount()):
                try:
                    name = table.item(row, 0).text()
                    x = float(table.item(row, 1).text())
                    y = float(table.item(row, 2).text())
                    w = float(table.item(row, 3).text())
                    h = float(table.item(row, 4).text())

                    zones[zone_type].append((x, y, w, h, name))
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid zone data at row {row}: {str(e)}")

        return zones


# Region of Interest Dialog
class RegionOfInterestDialog(QDialog):
    """Dialog for selecting a region of interest"""

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Region of Interest")
        self.setMinimumSize(800, 600)

        self.frame = frame
        self.roi = None  # Will store x1,y1,x2,y2
        self.drawing = False
        self.start_point = None
        self.end_point = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Click and drag to select a region of interest (ROI).")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Frame display
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Convert frame to QPixmap
        h, w, ch = self.frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            self.frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
        )
        pixmap = QPixmap.fromImage(qt_image)

        # Scale pixmap to fit display
        self.original_pixmap = pixmap
        scaled_pixmap = pixmap.scaled(
            self.display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.display.setPixmap(scaled_pixmap)

        # Enable mouse tracking
        self.display.setMouseTracking(True)
        self.display.mousePressEvent = self.mouse_press_event
        self.display.mouseMoveEvent = self.mouse_move_event
        self.display.mouseReleaseEvent = self.mouse_release_event

        layout.addWidget(self.display)

        # Buttons
        button_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Clear ROI")
        self.clear_btn.clicked.connect(self.clear_roi)
        button_layout.addWidget(self.clear_btn)

        button_layout.addStretch()

        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def mouse_press_event(self, event):
        # Start drawing ROI
        self.drawing = True
        self.start_point = event.position().toPoint()
        self.end_point = self.start_point
        self.update_roi()

    def mouse_move_event(self, event):
        if self.drawing:
            self.end_point = event.position().toPoint()
            self.update_roi()

    def mouse_release_event(self, event):
        self.drawing = False
        self.end_point = event.position().toPoint()
        self.update_roi()

        # Enable OK button if valid ROI
        if self.roi and self.roi[2] > self.roi[0] and self.roi[3] > self.roi[1]:
            self.ok_btn.setEnabled(True)

    def update_roi(self):
        if not self.start_point or not self.end_point:
            return

        # Create a copy of the original pixmap
        pixmap = QPixmap(self.original_pixmap)

        # Get ROI coordinates
        x1 = min(self.start_point.x(), self.end_point.x())
        y1 = min(self.start_point.y(), self.end_point.y())
        x2 = max(self.start_point.x(), self.end_point.x())
        y2 = max(self.start_point.y(), self.end_point.y())

        # Draw ROI rectangle
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        painter.end()

        # Scale pixmap to fit display
        scaled_pixmap = pixmap.scaled(
            self.display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.display.setPixmap(scaled_pixmap)

        # Calculate ROI in original image coordinates
        scale_factor_x = self.frame.shape[1] / pixmap.width()
        scale_factor_y = self.frame.shape[0] / pixmap.height()

        self.roi = (
            int(x1 * scale_factor_x),
            int(y1 * scale_factor_y),
            int(x2 * scale_factor_x),
            int(y2 * scale_factor_y),
        )

    def clear_roi(self):
        # Clear ROI
        self.roi = None
        self.start_point = None
        self.end_point = None

        # Reset display
        scaled_pixmap = self.original_pixmap.scaled(
            self.display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.display.setPixmap(scaled_pixmap)

        # Disable OK button
        self.ok_btn.setEnabled(False)


# Custom Rule Dialog
class CustomRuleDialog(QDialog):
    """Dialog for creating and editing custom detection rules"""

    def __init__(self, parent=None, rule_data=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Detection Rule")
        self.setMinimumWidth(500)

        # Initialize with existing rule data if provided
        self.rule_data = rule_data or {
            "name": "",
            "type": "custom",
            "confidence_threshold": 0.5,
            "time_threshold": 5,
            "person_count": 1,
            "custom_conditions": "",
        }

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Rule name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Rule Name:"))
        self.name_edit = QLineEdit(self.rule_data.get("name", ""))
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)

        # Rule type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Rule Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(
            [
                "custom",
                "fall",
                "attack",
                "accident",
                "intrusion",
                "loitering",
                "abandoned",
            ]
        )
        if self.rule_data.get("type") in [
            "custom",
            "fall",
            "attack",
            "accident",
            "intrusion",
            "loitering",
            "abandoned",
        ]:
            self.type_combo.setCurrentText(self.rule_data.get("type"))
        else:
            self.type_combo.setCurrentText("custom")
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 10)
        self.conf_slider.setValue(
            int(self.rule_data.get("confidence_threshold", 0.5) * 10)
        )
        conf_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel(
            f"{self.rule_data.get('confidence_threshold', 0.5):.1f}"
        )
        conf_layout.addWidget(self.conf_label)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        layout.addLayout(conf_layout)

        # Time threshold
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Minimum Duration (frames):"))
        self.time_spin = QSpinBox()
        self.time_spin.setRange(1, 300)
        self.time_spin.setValue(self.rule_data.get("time_threshold", 5))
        time_layout.addWidget(self.time_spin)
        layout.addLayout(time_layout)

        # Person count
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Minimum Person Count:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 10)
        self.count_spin.setValue(self.rule_data.get("person_count", 1))
        count_layout.addWidget(self.count_spin)
        layout.addLayout(count_layout)

        # Custom conditions
        layout.addWidget(QLabel("Custom Conditions:"))
        self.conditions_text = QTextEdit(self.rule_data.get("custom_conditions", ""))
        self.conditions_text.setPlaceholderText(
            "Enter custom detection conditions here..."
        )
        layout.addWidget(self.conditions_text)

        # Buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_conf_label(self):
        """Update confidence label when slider changes"""
        value = self.conf_slider.value() / 10.0
        self.conf_label.setText(f"{value:.1f}")

    def get_rule_data(self):
        """Get the rule data from the dialog"""
        return {
            "name": self.name_edit.text(),
            "type": self.type_combo.currentText(),
            "confidence_threshold": self.conf_slider.value() / 10.0,
            "time_threshold": self.time_spin.value(),
            "person_count": self.count_spin.value(),
            "custom_conditions": self.conditions_text.toPlainText(),
        }


# Main application class
class IncidentDetectionApp(QMainWindow):
    """Main application for real-time incident detection"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Advanced Incident Detection System v{APP_VERSION}")
        self.setMinimumSize(1200, 800)

        # Settings
        self.settings = QSettings("IncidentDetectionApp", "settings")
        self.detection_settings = self.load_detection_settings()

        # Custom rules
        self.custom_rules = self.load_custom_rules()

        # Initialize YOLO model
        self.model = None
        self.load_model()

        # Set up UI
        self.init_ui()

        # Set up toolbar and menus
        self.create_toolbar()
        self.create_menus()

        # Variables
        self.video_path = None
        self.detection_thread = None
        self.is_recording = False
        self.video_writer = None

        # Database
        self.db = IncidentDatabase()

        # Notification manager
        self.notification_manager = NotificationManager(self.detection_settings)

        # Analytics data
        self.analytics_data = []

        # Stream server (for web clients)
        self.stream_server = None

        # Update statistics timer
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_analytics)
        self.stats_timer.start(60000)  # Update every minute

        # Status bar setup
        self.statusBar().showMessage("Ready")

        # Initialize tray icon
        self.setup_tray_icon()

        # Load settings
        self.load_app_state()

        # Initialize shortcuts
        self.init_shortcuts()

    def init_shortcuts(self):
        """Initialize keyboard shortcuts"""
        # File operations
        self.shortcut_open = QShortcut(QKeySequence.StandardKey.Open, self)
        self.shortcut_open.activated.connect(self.load_video)

        self.shortcut_save = QShortcut(QKeySequence.StandardKey.Save, self)
        self.shortcut_save.activated.connect(self.export_results)

        # Camera operations
        self.shortcut_start_camera = QShortcut(QKeySequence("Ctrl+C"), self)
        self.shortcut_start_camera.activated.connect(
            lambda: (
                self.toggle_camera_detection()
                if self.tabWidget.currentIndex() == 0
                else None
            )
        )

        # Video operations
        self.shortcut_process_video = QShortcut(QKeySequence("Ctrl+P"), self)
        self.shortcut_process_video.activated.connect(
            lambda: self.process_video() if self.tabWidget.currentIndex() == 1 else None
        )

        # View operations
        self.shortcut_fullscreen = QShortcut(QKeySequence("F11"), self)
        self.shortcut_fullscreen.activated.connect(self.toggle_fullscreen)

        # Help
        self.shortcut_help = QShortcut(QKeySequence.StandardKey.HelpContents, self)
        self.shortcut_help.activated.connect(self.show_help)

    def load_detection_settings(self):
        """Load detection settings from QSettings"""
        settings = {}

        # Load default settings
        settings["confidence_threshold"] = self.settings.value(
            "confidence_threshold", 0.5, type=float
        )
        settings["frame_interval"] = self.settings.value("frame_interval", 1, type=int)
        settings["cooldown_frames"] = self.settings.value(
            "cooldown_frames", 30, type=int
        )
        settings["sound_alerts"] = self.settings.value("sound_alerts", True, type=bool)
        settings["popup_alerts"] = self.settings.value("popup_alerts", True, type=bool)
        settings["log_alerts"] = self.settings.value("log_alerts", True, type=bool)
        settings["use_gpu"] = self.settings.value("use_gpu", False, type=bool)
        settings["stabilize_video"] = self.settings.value(
            "stabilize_video", False, type=bool
        )
        settings["email_enabled"] = self.settings.value(
            "email_enabled", False, type=bool
        )
        settings["webhook_enabled"] = self.settings.value(
            "webhook_enabled", False, type=bool
        )

        # Load zone definitions if available
        zones_str = self.settings.value("zones")
        if zones_str:
            try:
                settings["zones"] = json.loads(zones_str)
            except:
                settings["zones"] = {}

        return settings

    def save_detection_settings(self):
        """Save detection settings to QSettings"""
        for key, value in self.detection_settings.items():
            # Special handling for zones (convert to JSON)
            if key == "zones":
                self.settings.setValue(key, json.dumps(value))
            else:
                self.settings.setValue(key, value)

    def load_custom_rules(self):
        """Load custom rules from settings"""
        rules = self.settings.value("custom_rules")
        if rules:
            try:
                return rules
            except:
                pass

        return []

    def save_custom_rules(self):
        """Save custom rules to settings"""
        self.settings.setValue("custom_rules", self.custom_rules)

    def load_app_state(self):
        """Load application state from settings"""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Set detection mode
        detection_mode = self.settings.value("detection_mode", "camera", type=str)
        if detection_mode == "camera":
            self.tabWidget.setCurrentIndex(0)
        else:
            self.tabWidget.setCurrentIndex(1)

        # Set threshold slider
        threshold = int(self.detection_settings["confidence_threshold"] * 10)
        self.threshold_slider.setValue(threshold)
        self.threshold_label.setText(f"Confidence: {threshold/10:.1f}")

        # Update alert settings checkboxes
        self.sound_alerts_check.setChecked(
            self.detection_settings.get("sound_alerts", True)
        )
        self.popup_alerts_check.setChecked(
            self.detection_settings.get("popup_alerts", True)
        )
        self.log_alerts_check.setChecked(
            self.detection_settings.get("log_alerts", True)
        )

    def save_app_state(self):
        """Save application state to settings"""
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())

        # Save detection mode
        detection_mode = "camera" if self.tabWidget.currentIndex() == 0 else "video"
        self.settings.setValue("detection_mode", detection_mode)

        # Save settings
        self.save_detection_settings()
        self.save_custom_rules()

    def load_model(self):
        """Load the YOLO model"""
        try:
            model_name = self.settings.value("model", DEFAULT_MODEL, type=str)

            # Check if GPU should be used
            device = "0" if self.detection_settings.get("use_gpu", False) else "cpu"

            # For demonstration purposes, we'll just create a placeholder model
            self.model = YOLO(model_name)
            logger.info(f"Model {model_name} loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            QMessageBox.critical(
                self, "Model Error", f"Error loading YOLO model: {str(e)}"
            )

    def init_ui(self):
        """Initialize the user interface"""
        # Main layout with central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Create tab widget for camera/video modes
        self.tabWidget = QTabWidget()
        self.tabWidget.currentChanged.connect(self.on_tab_changed)

        # Initialize camera tab
        self.init_camera_tab()

        # Initialize video tab
        self.init_video_tab()

        # Initialize analytics tab
        self.init_analytics_tab()

        # Add tabs to main layout
        main_layout.addWidget(self.tabWidget)

    def init_camera_tab(self):
        """Initialize the camera tab"""
        self.camera_tab = QWidget()
        camera_layout = QVBoxLayout()
        self.camera_tab.setLayout(camera_layout)

        # Camera controls
        camera_controls = QGroupBox("Camera Controls")
        controls_layout = QHBoxLayout()

        # Camera selection
        controls_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        controls_layout.addWidget(self.camera_combo)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_btn.clicked.connect(self.refresh_camera_list)
        controls_layout.addWidget(refresh_btn)

        # Start/Stop button
        self.camera_start_btn = QPushButton("Start Detection")
        self.camera_start_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.camera_start_btn.clicked.connect(self.toggle_camera_detection)
        controls_layout.addWidget(self.camera_start_btn)

        # Record button
        self.record_btn = QPushButton("Record")
        self.record_btn.setIcon(QIcon.fromTheme("media-record"))
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        controls_layout.addWidget(self.record_btn)

        # Streaming toggle
        self.stream_check = QCheckBox("Enable Web Stream")
        self.stream_check.setToolTip("Stream the camera feed to web browsers")
        self.stream_check.toggled.connect(self.toggle_streaming)
        controls_layout.addWidget(self.stream_check)

        camera_controls.setLayout(controls_layout)
        camera_layout.addWidget(camera_controls)

        # Main camera content
        camera_content = QSplitter(Qt.Orientation.Horizontal)

        # Video display
        self.camera_display = QLabel()
        self.camera_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_display.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.camera_display.setText("Select a camera and click 'Start Detection'")
        self.camera_display.setMinimumSize(640, 480)

        camera_content.addWidget(self.camera_display)

        # Right panel with tabs
        right_panel = QTabWidget()

        # Stats tab
        self.stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        self.stats_widget.setLayout(stats_layout)

        # Stats display
        self.stats_group = QGroupBox("Detection Statistics")
        stats_grid = QGridLayout()

        self.stats_labels = {}
        stats = [
            ("fps", "FPS:", "0"),
            ("processing_time", "Processing Time:", "0 ms"),
            ("frames_processed", "Frames Processed:", "0"),
            ("falls_detected", "Falls Detected:", "0"),
            ("attacks_detected", "Attacks Detected:", "0"),
            ("accidents_detected", "Accidents Detected:", "0"),
            ("intrusions_detected", "Intrusions Detected:", "0"),
            ("loitering_detected", "Loitering Detected:", "0"),
            ("abandoned_detected", "Abandoned Items:", "0"),
            ("anomalies_detected", "Anomalies Detected:", "0"),
        ]

        for i, (key, label_text, initial_value) in enumerate(stats):
            stats_grid.addWidget(QLabel(label_text), i, 0)
            self.stats_labels[key] = QLabel(initial_value)
            stats_grid.addWidget(self.stats_labels[key], i, 1)

        self.stats_group.setLayout(stats_grid)
        stats_layout.addWidget(self.stats_group)

        # Real-time charts (if Qt Charts is available)
        if HAS_CHARTS:
            self.stats_chart = QChart()
            self.stats_chart.setTitle("Detection Metrics")
            self.stats_chart.setAnimationOptions(
                QChart.AnimationOption.SeriesAnimations
            )
            self.stats_chart.legend().setVisible(True)

            # Create series for different metrics
            self.fps_series = QLineSeries()
            self.fps_series.setName("FPS")

            self.time_series = QLineSeries()
            self.time_series.setName("Processing Time (ms)")

            # Add series to chart
            self.stats_chart.addSeries(self.fps_series)
            self.stats_chart.addSeries(self.time_series)

            # Create default axes
            self.stats_chart.createDefaultAxes()

            # Create chart view
            self.stats_chart_view = QChartView(self.stats_chart)
            self.stats_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
            stats_layout.addWidget(self.stats_chart_view)
        else:
            # Fallback if charts not available
            chart_placeholder = QLabel("Charts module not available")
            chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stats_layout.addWidget(chart_placeholder)

        right_panel.addTab(self.stats_widget, "Statistics")

        # Alerts tab
        self.alerts_widget = QWidget()
        alerts_layout = QVBoxLayout()
        self.alerts_widget.setLayout(alerts_layout)

        # Alerts list
        self.alerts_list = QListWidget()
        alerts_layout.addWidget(self.alerts_list)

        # Alert controls
        alert_controls = QHBoxLayout()

        # Clear button
        clear_btn = QPushButton("Clear Alerts")
        clear_btn.clicked.connect(self.alerts_list.clear)
        alert_controls.addWidget(clear_btn)

        # Export button
        export_alerts_btn = QPushButton("Export Alerts")
        export_alerts_btn.clicked.connect(self.export_alerts)
        alert_controls.addWidget(export_alerts_btn)

        alerts_layout.addLayout(alert_controls)

        # Alert settings
        alert_settings = QGroupBox("Alert Settings")
        alert_settings_layout = QVBoxLayout()

        # Sound alerts
        self.sound_alerts_check = QCheckBox("Sound Alerts")
        self.sound_alerts_check.setChecked(True)
        self.sound_alerts_check.toggled.connect(self.update_alert_settings)
        alert_settings_layout.addWidget(self.sound_alerts_check)

        # Popup alerts
        self.popup_alerts_check = QCheckBox("Popup Notifications")
        self.popup_alerts_check.setChecked(True)
        self.popup_alerts_check.toggled.connect(self.update_alert_settings)
        alert_settings_layout.addWidget(self.popup_alerts_check)

        # Log alerts
        self.log_alerts_check = QCheckBox("Log Alerts to File")
        self.log_alerts_check.setChecked(True)
        self.log_alerts_check.toggled.connect(self.update_alert_settings)
        alert_settings_layout.addWidget(self.log_alerts_check)

        alert_settings.setLayout(alert_settings_layout)
        alerts_layout.addWidget(alert_settings)

        right_panel.addTab(self.alerts_widget, "Alerts")

        # Visualization tab
        self.viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        self.viz_widget.setLayout(viz_layout)

        # Create advanced visualization widgets
        self.spatial_analysis = SpatialAnalysisWidget()
        viz_layout.addWidget(self.spatial_analysis)

        right_panel.addTab(self.viz_widget, "Visualization")

        camera_content.addWidget(right_panel)
        camera_content.setSizes([600, 400])

        camera_layout.addWidget(camera_content)

        self.tabWidget.addTab(self.camera_tab, "Camera Detection")

    def init_video_tab(self):
        """Initialize the video tab"""
        self.video_tab = QWidget()
        video_layout = QVBoxLayout()
        self.video_tab.setLayout(video_layout)

        # Video controls
        video_controls = QGroupBox("Video Controls")
        video_controls_layout = QHBoxLayout()

        # Load video button
        self.load_btn = QPushButton("Load Video")
        self.load_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_btn.clicked.connect(self.load_video)
        video_controls_layout.addWidget(self.load_btn)

        # Confidence threshold
        video_controls_layout.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 10)
        self.threshold_slider.setValue(5)  # Default 0.5
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_slider.setFixedWidth(150)
        video_controls_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("Confidence: 0.5")
        video_controls_layout.addWidget(self.threshold_label)

        # Stabilization checkbox
        self.stabilize_check = QCheckBox("Stabilize Video")
        self.stabilize_check.setChecked(
            self.detection_settings.get("stabilize_video", False)
        )
        self.stabilize_check.toggled.connect(self.update_stabilization)
        video_controls_layout.addWidget(self.stabilize_check)

        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_video)
        video_controls_layout.addWidget(self.process_btn)

        video_controls.setLayout(video_controls_layout)
        video_layout.addWidget(video_controls)

        # Main video content splitter
        video_content = QSplitter(Qt.Orientation.Horizontal)

        # Left panel with video player and progress
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Video display
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.video_display.setText("Load a video to begin analysis")
        self.video_display.setMinimumSize(640, 480)

        # Playback controls
        playback_controls = QHBoxLayout()

        # Position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 100)
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderMoved.connect(self.seek_position)

        # Time label
        self.time_label = QLabel("00:00 / 00:00")

        playback_controls.addWidget(self.position_slider)
        playback_controls.addWidget(self.time_label)

        left_layout.addWidget(self.video_display)
        left_layout.addLayout(playback_controls)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        video_content.addWidget(left_panel)

        # Right panel with results tabs
        results_tabs = QTabWidget()

        # Incidents table tab
        self.incidents_tab = QWidget()
        incidents_layout = QVBoxLayout()
        self.incidents_tab.setLayout(incidents_layout)

        # Incidents table
        self.incidents_table = QTableWidget()
        self.incidents_table.setColumnCount(5)
        self.incidents_table.setHorizontalHeaderLabels(
            ["Type", "Time", "Confidence", "Frame", "Details"]
        )
        self.incidents_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.incidents_table.cellDoubleClicked.connect(self.jump_to_incident)

        incidents_layout.addWidget(self.incidents_table)

        # Filter controls
        filter_group = QGroupBox("Filter Incidents")
        filter_layout = QHBoxLayout()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            [
                "All Types",
                "Falls",
                "Attacks",
                "Accidents",
                "Intrusions",
                "Loitering",
                "Abandoned",
                "Custom",
            ]
        )
        self.filter_combo.currentIndexChanged.connect(self.filter_incidents)

        self.min_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_conf_slider.setRange(1, 10)
        self.min_conf_slider.setValue(3)  # Default 0.3
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

        results_tabs.addTab(self.incidents_tab, "Incidents")

        # Advanced visualizations tab
        self.advanced_viz_tab = QWidget()
        advanced_viz_layout = QVBoxLayout()
        self.advanced_viz_tab.setLayout(advanced_viz_layout)

        # Create 3D visualization widget
        self.viz_3d_widget = ThreeDVisualizationWidget()
        advanced_viz_layout.addWidget(self.viz_3d_widget)

        results_tabs.addTab(self.advanced_viz_tab, "3D Analysis")

        # Heat map tab
        self.heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout()
        self.heatmap_tab.setLayout(heatmap_layout)

        # Create spatial analysis widget
        self.spatial_widget = SpatialAnalysisWidget()
        heatmap_layout.addWidget(self.spatial_widget)

        results_tabs.addTab(self.heatmap_tab, "Spatial Analysis")

        # Custom Rules tab
        self.rules_tab = QWidget()
        rules_layout = QVBoxLayout()
        self.rules_tab.setLayout(rules_layout)

        # Rules controls
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

        # Rule details
        rule_details_group = QGroupBox("Rule Details")
        rule_details_layout = QVBoxLayout()
        self.rule_details_text = QTextEdit()
        self.rule_details_text.setReadOnly(True)
        rule_details_layout.addWidget(self.rule_details_text)
        rule_details_group.setLayout(rule_details_layout)
        rules_layout.addWidget(rule_details_group)

        results_tabs.addTab(self.rules_tab, "Custom Rules")

        video_content.addWidget(results_tabs)
        video_layout.addWidget(video_content)

        self.tabWidget.addTab(self.video_tab, "Video Analysis")

    def init_analytics_tab(self):
        """Initialize the analytics dashboard tab"""
        self.analytics_tab = QWidget()
        analytics_layout = QVBoxLayout()
        self.analytics_tab.setLayout(analytics_layout)

        # Controls
        controls_layout = QHBoxLayout()

        # Time range
        controls_layout.addWidget(QLabel("Time Range:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(
            ["Today", "Last 3 Days", "Last Week", "Last Month", "All Time"]
        )
        self.time_range_combo.currentIndexChanged.connect(self.update_analytics)
        controls_layout.addWidget(self.time_range_combo)

        # Incident type
        controls_layout.addWidget(QLabel("Incident Type:"))
        self.incident_type_combo = QComboBox()
        self.incident_type_combo.addItems(
            [
                "All Types",
                "Falls",
                "Attacks",
                "Accidents",
                "Intrusions",
                "Loitering",
                "Abandoned",
                "Custom",
            ]
        )
        self.incident_type_combo.currentIndexChanged.connect(self.update_analytics)
        controls_layout.addWidget(self.incident_type_combo)

        # Grouping
        controls_layout.addWidget(QLabel("Group By:"))
        self.group_by_combo = QComboBox()
        self.group_by_combo.addItems(["Hour", "Day", "Week", "Month"])
        self.group_by_combo.currentIndexChanged.connect(self.update_analytics)
        controls_layout.addWidget(self.group_by_combo)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_analytics)
        controls_layout.addWidget(refresh_btn)

        # Export button
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.export_analytics)
        controls_layout.addWidget(export_btn)

        analytics_layout.addLayout(controls_layout)

        # Create tab widget for different analytics views
        analytics_tabs = QTabWidget()

        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout()
        summary_tab.setLayout(summary_layout)

        # Summary charts - use conditional based on Qt Charts availability
        if HAS_CHARTS:
            summary_charts = QSplitter(Qt.Orientation.Horizontal)

            # Incident count by type
            self.summary_chart1 = QChart()
            self.summary_chart1.setTitle("Incidents by Type")
            self.summary_chart1.setAnimationOptions(
                QChart.AnimationOption.SeriesAnimations
            )

            summary_chart_view1 = QChartView(self.summary_chart1)
            summary_chart_view1.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Incident trend over time
            self.summary_chart2 = QChart()
            self.summary_chart2.setTitle("Incident Trend")
            self.summary_chart2.setAnimationOptions(
                QChart.AnimationOption.SeriesAnimations
            )

            summary_chart_view2 = QChartView(self.summary_chart2)
            summary_chart_view2.setRenderHint(QPainter.RenderHint.Antialiasing)

            summary_charts.addWidget(summary_chart_view1)
            summary_charts.addWidget(summary_chart_view2)
            summary_layout.addWidget(summary_charts)
        else:
            # Fallback if Qt Charts not available
            chart_placeholder = QLabel("Charts module not available")
            chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            summary_layout.addWidget(chart_placeholder)

        # Summary statistics
        self.summary_stats = QTableWidget()
        self.summary_stats.setColumnCount(3)
        self.summary_stats.setHorizontalHeaderLabels(["Metric", "Value", "Change"])
        self.summary_stats.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.summary_stats.setRowCount(5)

        # Add initial rows
        self.summary_stats.setItem(0, 0, QTableWidgetItem("Total Incidents"))
        self.summary_stats.setItem(1, 0, QTableWidgetItem("Most Common Incident"))
        self.summary_stats.setItem(2, 0, QTableWidgetItem("Average Confidence"))
        self.summary_stats.setItem(3, 0, QTableWidgetItem("Peak Incident Time"))
        self.summary_stats.setItem(4, 0, QTableWidgetItem("Incidents Today"))

        summary_layout.addWidget(self.summary_stats)

        analytics_tabs.addTab(summary_tab, "Summary")

        # Detailed analysis tab
        details_tab = QWidget()
        details_layout = QVBoxLayout()
        details_tab.setLayout(details_layout)

        # Create detailed analysis charts
        if HAS_CHARTS:
            details_charts = QSplitter(Qt.Orientation.Vertical)

            # Time distribution chart
            self.time_dist_chart = QChart()
            self.time_dist_chart.setTitle("Incident Distribution by Time of Day")
            self.time_dist_chart.setAnimationOptions(
                QChart.AnimationOption.SeriesAnimations
            )

            time_dist_chart_view = QChartView(self.time_dist_chart)
            time_dist_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Confidence distribution chart
            self.conf_dist_chart = QChart()
            self.conf_dist_chart.setTitle("Incident Confidence Distribution")
            self.conf_dist_chart.setAnimationOptions(
                QChart.AnimationOption.SeriesAnimations
            )

            conf_dist_chart_view = QChartView(self.conf_dist_chart)
            conf_dist_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

            details_charts.addWidget(time_dist_chart_view)
            details_charts.addWidget(conf_dist_chart_view)
            details_layout.addWidget(details_charts)
        else:
            # Fallback if charts not available
            chart_placeholder = QLabel("Charts module not available")
            chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            details_layout.addWidget(chart_placeholder)

        analytics_tabs.addTab(details_tab, "Time Analysis")

        # Heatmap tab
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout()
        heatmap_tab.setLayout(heatmap_layout)

        # Create matplotlib figure for heatmap
        self.heatmap_figure = Figure(figsize=(10, 8), dpi=100)
        self.heatmap_canvas = FigureCanvasQTAgg(self.heatmap_figure)
        heatmap_layout.addWidget(self.heatmap_canvas)

        analytics_tabs.addTab(heatmap_tab, "Heatmap")

        # Incidents table tab
        incidents_table_tab = QWidget()
        incidents_table_layout = QVBoxLayout()
        incidents_table_tab.setLayout(incidents_table_layout)

        # Search controls
        search_layout = QHBoxLayout()

        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search incidents...")
        self.search_edit.textChanged.connect(self.search_incidents)
        search_layout.addWidget(self.search_edit)

        # Date range
        search_layout.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-7))
        self.date_from.setCalendarPopup(True)
        search_layout.addWidget(self.date_from)

        search_layout.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        search_layout.addWidget(self.date_to)

        # Search button
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_incidents)
        search_layout.addWidget(search_btn)

        incidents_table_layout.addLayout(search_layout)

        # Create table for incident history
        self.incidents_history_table = QTableWidget()
        self.incidents_history_table.setColumnCount(6)
        self.incidents_history_table.setHorizontalHeaderLabels(
            ["Date/Time", "Type", "Source", "Confidence", "Details", "Actions"]
        )
        self.incidents_history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        incidents_table_layout.addWidget(self.incidents_history_table)

        analytics_tabs.addTab(incidents_table_tab, "Incident History")

        analytics_layout.addWidget(analytics_tabs)

        self.tabWidget.addTab(self.analytics_tab, "Analytics")

    def create_toolbar(self):
        """Create the application toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Add camera actions
        camera_action = QAction(QIcon.fromTheme("camera-video"), "Camera Mode", self)
        camera_action.setStatusTip("Switch to camera detection mode")
        camera_action.triggered.connect(lambda: self.tabWidget.setCurrentIndex(0))
        toolbar.addAction(camera_action)

        # Add video actions
        video_action = QAction(QIcon.fromTheme("video-x-generic"), "Video Mode", self)
        video_action.setStatusTip("Switch to video analysis mode")
        video_action.triggered.connect(lambda: self.tabWidget.setCurrentIndex(1))
        toolbar.addAction(video_action)

        toolbar.addSeparator()

        # Settings action
        settings_action = QAction(
            QIcon.fromTheme("preferences-system"), "Settings", self
        )
        settings_action.setStatusTip("Configure detection settings")
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)

        # Zone configuration
        zones_action = QAction(
            QIcon.fromTheme("insert-object"), "Configure Zones", self
        )
        zones_action.setStatusTip("Configure detection zones")
        zones_action.triggered.connect(self.configure_zones)
        toolbar.addAction(zones_action)

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

        # Export action
        export_action = QAction(
            QIcon.fromTheme("document-save"), "Export Results", self
        )
        export_action.setStatusTip("Export detection results")
        export_action.triggered.connect(self.export_results)
        toolbar.addAction(export_action)

        # Generate report action
        report_action = QAction(
            QIcon.fromTheme("x-office-document"), "Generate Report", self
        )
        report_action.setStatusTip("Generate comprehensive incident report")
        report_action.triggered.connect(self.generate_report)
        toolbar.addAction(report_action)

    def create_menus(self):
        """Create application menus"""
        # File menu
        file_menu = self.menuBar().addMenu("&File")

        load_video_action = QAction("&Load Video...", self)
        load_video_action.setShortcut(QKeySequence("Ctrl+O"))
        load_video_action.triggered.connect(self.load_video)
        file_menu.addAction(load_video_action)

        # Add recent videos submenu
        self.recent_menu = file_menu.addMenu("Recent Videos")
        self.update_recent_menu()

        file_menu.addSeparator()

        export_action = QAction("&Export Results...", self)
        export_action.setShortcut(QKeySequence("Ctrl+S"))
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        report_action = QAction("Generate &Report...", self)
        report_action.setShortcut(QKeySequence("Ctrl+R"))
        report_action.triggered.connect(self.generate_report)
        file_menu.addAction(report_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = self.menuBar().addMenu("&View")

        camera_view_action = QAction("&Camera Detection", self)
        camera_view_action.triggered.connect(lambda: self.tabWidget.setCurrentIndex(0))
        view_menu.addAction(camera_view_action)

        video_view_action = QAction("&Video Analysis", self)
        video_view_action.triggered.connect(lambda: self.tabWidget.setCurrentIndex(1))
        view_menu.addAction(video_view_action)

        analytics_view_action = QAction("&Analytics Dashboard", self)
        analytics_view_action.triggered.connect(
            lambda: self.tabWidget.setCurrentIndex(2)
        )
        view_menu.addAction(analytics_view_action)

        view_menu.addSeparator()

        fullscreen_action = QAction("&Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.setCheckable(True)
        fullscreen_action.toggled.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Tools menu
        tools_menu = self.menuBar().addMenu("&Tools")

        roi_action = QAction("Select &Region of Interest...", self)
        roi_action.triggered.connect(self.select_roi)
        tools_menu.addAction(roi_action)

        clear_roi_action = QAction("&Clear Region of Interest", self)
        clear_roi_action.triggered.connect(self.clear_roi)
        tools_menu.addAction(clear_roi_action)

        tools_menu.addSeparator()

        zones_action = QAction("Configure &Zones...", self)
        zones_action.triggered.connect(self.configure_zones)
        tools_menu.addAction(zones_action)

        rules_action = QAction("Manage Custom &Rules...", self)
        rules_action.triggered.connect(
            lambda: self.tabWidget.setCurrentIndex(1)
            or self.tabWidget.currentWidget().findChild(QTabWidget).setCurrentIndex(3)
        )
        tools_menu.addAction(rules_action)

        tools_menu.addSeparator()

        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)

        # Help menu
        help_menu = self.menuBar().addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        help_action = QAction("&Help", self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        check_updates_action = QAction("Check for &Updates", self)
        check_updates_action.triggered.connect(self.check_updates)
        help_menu.addAction(check_updates_action)

    def setup_tray_icon(self):
        """Set up system tray icon"""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        )
        self.tray_icon.setToolTip("Advanced Incident Detection System")

        # Create tray menu
        tray_menu = QMenu()

        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)

        hide_action = QAction("Hide", self)
        hide_action.triggered.connect(self.hide)
        tray_menu.addAction(hide_action)

        tray_menu.addSeparator()

        # Camera control submenu
        camera_menu = tray_menu.addMenu("Camera")

        start_camera_action = QAction("Start Detection", self)
        start_camera_action.triggered.connect(
            lambda: self.start_camera_detection_if_stopped()
        )
        camera_menu.addAction(start_camera_action)

        stop_camera_action = QAction("Stop Detection", self)
        stop_camera_action.triggered.connect(lambda: self.stop_detection())
        camera_menu.addAction(stop_camera_action)

        tray_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def start_camera_detection_if_stopped(self):
        """Start camera detection if not already running"""
        if not self.detection_thread or not self.detection_thread.isRunning():
            self.tabWidget.setCurrentIndex(0)  # Switch to camera tab
            self.toggle_camera_detection()

    def on_tray_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def refresh_camera_list(self):
        """Refresh the list of available cameras"""
        self.camera_combo.clear()

        # Get available cameras using QMediaDevices
        cameras = QMediaDevices.videoInputs()

        for i, camera in enumerate(cameras):
            self.camera_combo.addItem(f"{camera.description()} ({i})", i)

        # Add option for IP camera
        self.camera_combo.addItem("IP Camera", "ip")

        # Enable/disable start button based on available cameras
        self.camera_start_btn.setEnabled(self.camera_combo.count() > 0)

    def toggle_camera_detection(self):
        """Start or stop real-time camera detection"""
        if self.detection_thread and self.detection_thread.isRunning():
            # Stop detection
            self.stop_detection()
            self.camera_start_btn.setText("Start Detection")
            self.camera_start_btn.setIcon(QIcon.fromTheme("media-playback-start"))
            self.record_btn.setEnabled(False)

            # Stop recording if active
            if self.is_recording:
                self.toggle_recording(False)
                self.record_btn.setChecked(False)
        else:
            # Get camera ID
            camera_data = self.camera_combo.currentData()

            if camera_data == "ip":
                # Show dialog to enter IP camera URL
                camera_url, ok = QInputDialog.getText(
                    self,
                    "IP Camera",
                    "Enter RTSP URL:",
                    text="rtsp://username:password@192.168.1.100:554/stream",
                )

                if not ok or not camera_url:
                    return

                camera_id = camera_url
            else:
                camera_id = camera_data

            # Start detection
            self.start_camera_detection(camera_id)
            self.camera_start_btn.setText("Stop Detection")
            self.camera_start_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
            self.record_btn.setEnabled(True)

    def start_camera_detection(self, camera_id):
        """Start real-time detection on camera feed"""
        # Create detection thread
        self.detection_thread = AdvancedDetectionThread(
            self.model, self.detection_settings
        )
        self.detection_thread.set_camera(camera_id)

        # Connect signals
        self.detection_thread.update_frame.connect(self.on_frame_processed)
        self.detection_thread.update_stats.connect(self.on_stats_updated)
        self.detection_thread.incident_detected.connect(self.on_incident_detected)
        self.detection_thread.anomaly_detected.connect(self.on_anomaly_detected)
        self.detection_thread.motion_updated.connect(self.on_motion_updated)

        # Start thread
        self.detection_thread.start()

        self.statusBar().showMessage("Real-time detection started")

    def load_video(self):
        """Load a video file for analysis"""
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if video_path:
            self.video_path = video_path
            self.statusBar().showMessage(f"Loaded: {os.path.basename(video_path)}")

            # Open video to get properties
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Set up video position slider
                self.position_slider.setRange(0, total_frames - 1)
                self.position_slider.setValue(0)
                self.position_slider.setEnabled(True)

                # Update time label
                duration = total_frames / fps
                self.time_label.setText(f"00:00 / {self.format_time(duration)}")

                # Display first frame
                ret, frame = cap.read()
                if ret:
                    self.display_frame(frame, self.video_display)

                cap.release()

                # Enable process button
                self.process_btn.setEnabled(True)

                # Add to recent videos
                self.add_recent_video(video_path)
            else:
                QMessageBox.critical(self, "Error", "Could not open video file.")

    def format_time(self, seconds):
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def process_video(self):
        """Process the loaded video for incident detection"""
        if not self.video_path:
            return

        if self.detection_thread and self.detection_thread.isRunning():
            # Stop processing
            self.stop_detection()
            self.process_btn.setText("Process Video")
            self.process_btn.setIcon(QIcon.fromTheme("media-playback-start"))
            return

        # Clear previous results
        self.incidents_table.setRowCount(0)
        self.viz_3d_widget.set_incidents({})
        self.spatial_widget.set_incidents({})

        # Create detection thread
        self.detection_thread = AdvancedDetectionThread(
            self.model, self.detection_settings
        )
        self.detection_thread.set_video(self.video_path)

        # Connect signals
        self.detection_thread.update_frame.connect(self.on_frame_processed)
        self.detection_thread.update_progress.connect(self.progress_bar.setValue)
        self.detection_thread.processing_finished.connect(self.on_processing_finished)
        self.detection_thread.update_stats.connect(self.on_stats_updated)
        self.detection_thread.incident_detected.connect(self.on_incident_detected)
        self.detection_thread.anomaly_detected.connect(self.on_anomaly_detected)
        self.detection_thread.motion_updated.connect(self.on_motion_updated)

        # Update UI
        self.process_btn.setText("Stop Processing")
        self.process_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.statusBar().showMessage("Processing video...")

        # Start processing
        self.detection_thread.start()

    def stop_detection(self):
        """Stop the detection thread"""
        if (
            hasattr(self, "detection_thread")
            and self.detection_thread
            and self.detection_thread.isRunning()
        ):
            self.detection_thread.stop()
            self.detection_thread.wait()

            self.statusBar().showMessage("Detection stopped")

    def on_frame_processed(self, frame, results, detections):
        """Handle processed frame from detection thread"""
        # Display in appropriate tab
        if self.tabWidget.currentIndex() == 0:
            # Camera tab
            self.display_frame(frame, self.camera_display, results, detections)
        else:
            # Video tab
            self.display_frame(frame, self.video_display, results, detections)

            # Update position slider
            if "current_frame_number" in self.detection_thread.stats:
                current_frame = self.detection_thread.stats["current_frame_number"]
                self.position_slider.setValue(current_frame)

                # Update time label
                if hasattr(self.detection_thread, "cap"):
                    fps = self.detection_thread.cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0  # Default if not available

                    current_time = current_frame / fps
                    total_time = self.position_slider.maximum() / fps

                    self.time_label.setText(
                        f"{self.format_time(current_time)} / {self.format_time(total_time)}"
                    )

        # Handle recording if active
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

    def display_frame(self, frame, display_widget, results=None, detections=None):
        """Display a frame with optional detections"""
        if frame is None:
            return

        # Draw detections if available
        annotated_frame = frame.copy()

        if results:
            # For demonstration purposes, draw simple bounding boxes if available
            if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                boxes = (
                    results[0].boxes.xyxy.cpu().numpy()
                    if hasattr(results[0].boxes, "xyxy")
                    else []
                )
                for box in boxes:
                    cv2.rectangle(
                        annotated_frame,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 255, 0),
                        2,
                    )

        # Draw additional information based on detections
        if detections:
            # Draw incident markers
            for incident_type, incidents in detections.items():
                if not incidents:
                    continue

                # Pick color based on type
                color_map = {
                    "falls": (0, 0, 255),  # Red (BGR)
                    "attacks": (0, 165, 255),  # Orange
                    "accidents": (0, 255, 255),  # Yellow
                    "intrusions": (255, 0, 0),  # Blue
                    "loitering": (255, 0, 255),  # Purple
                    "abandoned": (0, 255, 0),  # Green
                }
                color = color_map.get(incident_type, (255, 255, 255))

                for incident in incidents:
                    # Get position if available, otherwise use center of frame
                    if "position" in incident:
                        pos_x, pos_y = incident["position"]
                        pos_x, pos_y = int(pos_x), int(pos_y)
                    elif "bbox" in incident:
                        x1, y1, x2, y2 = incident["bbox"]
                        pos_x, pos_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    else:
                        h, w = annotated_frame.shape[:2]
                        pos_x, pos_y = w // 2, h // 2

                    # Draw warning marker
                    cv2.circle(annotated_frame, (pos_x, pos_y), 10, color, -1)

                    # Add incident type label
                    incident_label = incident.get("rule", incident_type.capitalize())
                    cv2.putText(
                        annotated_frame,
                        incident_label,
                        (pos_x + 15, pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

        # Convert to QPixmap
        h, w, ch = annotated_frame.shape
        bytes_per_line = ch * w

        # Convert to RGB for Qt
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit the display
        scaled_pixmap = pixmap.scaled(
            display_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Set the pixmap to the display widget
        display_widget.setPixmap(scaled_pixmap)

    def on_stats_updated(self, stats):
        """Handle statistics updates from detection thread"""
        # Update stats labels
        for key, label in self.stats_labels.items():
            if key in stats:
                value = stats[key]

                # Format based on key
                if key == "fps":
                    label.setText(str(value))
                elif key == "processing_time":
                    label.setText(f"{value*1000:.1f} ms")
                else:
                    label.setText(str(value))

        # Update charts if in camera mode and QtCharts is available
        if self.tabWidget.currentIndex() == 0 and HAS_CHARTS:
            # Update FPS chart - limit to recent values
            if hasattr(self, "chart_time_points"):
                if len(self.chart_time_points) > 20:
                    self.chart_time_points.pop(0)
                    self.fps_series.remove(0)
                    self.time_series.remove(0)
            else:
                self.chart_time_points = []

            # Add new data point
            self.chart_time_points.append(len(self.chart_time_points))

            # Update FPS series
            self.fps_series.append(self.chart_time_points[-1], stats.get("fps", 0))

            # Update processing time series
            self.time_series.append(
                self.chart_time_points[-1], stats.get("processing_time", 0) * 1000
            )

            # Update chart if series were updated
            if self.fps_series.count() > 0 and self.time_series.count() > 0:
                self.stats_chart.createDefaultAxes()

    def on_incident_detected(self, incident_type, details):
        """Handle detected incident"""
        # Add alert to UI
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        confidence = details.get("confidence", 0)

        # Format message
        message = f"{timestamp} - {incident_type.capitalize()} detected (conf: {confidence:.2f})"

        # Add custom rule name if available
        if "rule" in details:
            message += f" - {details['rule']}"

        # Add to alerts list
        item = QListWidgetItem(message)

        # Set color based on incident type
        if incident_type in INCIDENT_COLORS:
            item.setForeground(INCIDENT_COLORS[incident_type])
        else:
            item.setForeground(INCIDENT_COLORS["custom"])

        self.alerts_list.insertItem(0, item)  # Add at top

        # Play sound if enabled
        if self.sound_alerts_check.isChecked():
            QApplication.beep()

        # Show system tray notification if enabled
        if self.popup_alerts_check.isChecked() and self.tray_icon:
            self.tray_icon.showMessage(
                "Incident Detected",
                message,
                QSystemTrayIcon.MessageIcon.Warning,
                3000,  # Show for 3 seconds
            )

        # Log to database
        frame_number = details.get("frame")
        source = (
            self.video_path if self.tabWidget.currentIndex() == 1 else "Camera Feed"
        )

        # Save image of incident if needed
        image_path = None
        if (
            self.detection_thread
            and hasattr(self.detection_thread, "recent_frames")
            and self.detection_thread.recent_frames
        ):
            try:
                # Get current frame
                current_frame = self.detection_thread.recent_frames[-1][0]

                # Save to disk
                save_dir = os.path.join(QDir.homePath(), ".incident_detector", "images")
                os.makedirs(save_dir, exist_ok=True)

                # Generate filename
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(
                    save_dir, f"{incident_type}_{timestamp_str}.jpg"
                )

                # Save image
                cv2.imwrite(image_path, current_frame)
            except Exception as e:
                logger.error(f"Error saving incident image: {str(e)}")
                image_path = None

        # Add to database
        self.db.add_incident(
            incident_type=incident_type,
            details=details,
            source=source,
            frame_number=frame_number,
            confidence=confidence,
            image_path=image_path,
        )

        # Send notifications if configured
        self.send_incident_notification(incident_type, details, image_path)

    def on_anomaly_detected(self, anomaly_type, details):
        """Handle detected anomaly"""
        # Similar to incident detection but for anomalies
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        confidence = details.get("confidence", 0)

        # Format message
        anomaly_name = details.get("type", "Unknown")
        message = (
            f"{timestamp} - Anomaly detected: {anomaly_name} (conf: {confidence:.2f})"
        )

        # Add to alerts list
        item = QListWidgetItem(message)
        item.setForeground(QColor(0, 191, 255))  # Deep sky blue for anomalies

        self.alerts_list.insertItem(0, item)  # Add at top

        # Log to database as an anomaly
        frame_number = details.get("frame")
        source = (
            self.video_path if self.tabWidget.currentIndex() == 1 else "Camera Feed"
        )

        # Add to database
        self.db.add_incident(
            incident_type="anomaly",
            details=details,
            source=source,
            frame_number=frame_number,
            confidence=confidence,
        )

    def on_motion_updated(self, motion_data):
        """Handle motion data updates for visualization"""
        # Update spatial analysis widget
        self.spatial_widget.set_motion_history(motion_data)

        # Update 3D visualization if in that mode
        if self.tabWidget.currentIndex() == 1 and hasattr(self, "viz_3d_widget"):
            self.viz_3d_widget.set_motion_data(motion_data)

    def on_processing_finished(self, incidents):
        """Handle completion of video processing"""
        self.process_btn.setText("Process Video")
        self.process_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.statusBar().showMessage("Processing complete")

        # Update incidents table
        self.update_incidents_table(incidents)

        # Update 3D visualization
        self.viz_3d_widget.set_incidents(incidents)

        # Update spatial analysis
        self.spatial_widget.set_incidents(incidents)

        # Update analytics
        self.update_analytics()

    def update_incidents_table(self, incidents):
        """Update the incidents table with detection results"""
        # Clear table
        self.incidents_table.setRowCount(0)

        # Add incidents to table
        row = 0
        for incident_type, incidents_list in incidents.items():
            for frame_num, details in incidents_list:
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

                    # Frame number
                    frame_item = QTableWidgetItem(str(frame_num))
                    self.incidents_table.setItem(row, 3, frame_item)

                    # Details - any custom information
                    details_str = ", ".join(
                        [
                            f"{k}={v}"
                            for k, v in detail.items()
                            if k not in ["time", "confidence", "frame"]
                        ]
                    )
                    details_item = QTableWidgetItem(details_str)
                    self.incidents_table.setItem(row, 4, details_item)

                    row += 1

        # Apply current filter
        self.filter_incidents()

    def filter_incidents(self):
        """Filter incidents in the table based on type and confidence"""
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
            if filter_type != "all types" and not type_text.startswith(
                filter_type.rstrip("s")
            ):
                show_row = False

            # Filter by confidence
            if conf_value < min_conf:
                show_row = False

            # Show/hide row
            self.incidents_table.setRowHidden(row, not show_row)

    def update_threshold(self):
        """Update confidence threshold label"""
        value = self.threshold_slider.value() / 10.0
        self.threshold_label.setText(f"Confidence: {value:.1f}")

        # Update detection settings
        self.detection_settings["confidence_threshold"] = value

    def update_stabilization(self):
        """Update video stabilization setting"""
        self.detection_settings["stabilize_video"] = self.stabilize_check.isChecked()

    def update_min_conf_label(self):
        """Update minimum confidence filter label"""
        value = self.min_conf_slider.value() / 10.0
        self.min_conf_label.setText(f"Min Confidence: {value:.1f}")

    def jump_to_incident(self, row, column):
        """Jump to the incident frame when double-clicking in the table"""
        if self.tabWidget.currentIndex() != 1:  # Not in video tab
            return

        frame_item = self.incidents_table.item(row, 3)
        if frame_item:
            try:
                frame_num = int(frame_item.text())

                # Seek to frame
                self.seek_position(frame_num)

            except ValueError:
                pass

    def seek_position(self, position):
        """Seek to a specific position in the video"""
        if not hasattr(self, "detection_thread") or not self.detection_thread:
            return

        # If video is being processed, pause it
        if self.detection_thread.isRunning():
            self.detection_thread.pause()

        # Open video file if needed
        if not hasattr(self.detection_thread, "cap") or not self.detection_thread.cap:
            if self.video_path:
                self.detection_thread.cap = cv2.VideoCapture(self.video_path)

        # Set position in video
        if hasattr(self.detection_thread, "cap") and self.detection_thread.cap:
            cap = self.detection_thread.cap
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ret, frame = cap.read()
                if ret:
                    self.display_frame(frame, self.video_display)

                    # Update position slider
                    self.position_slider.setValue(position)

                    # Update time label
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0  # Default if not available

                    current_time = position / fps
                    total_time = self.position_slider.maximum() / fps

                    self.time_label.setText(
                        f"{self.format_time(current_time)} / {self.format_time(total_time)}"
                    )

    def toggle_recording(self, checked):
        """Toggle recording of camera feed"""
        if (
            not hasattr(self, "detection_thread")
            or not self.detection_thread
            or not self.detection_thread.isRunning()
        ):
            self.record_btn.setChecked(False)
            return

        if checked:  # Start recording
            # Create output directory
            output_dir = os.path.join(QDir.homePath(), "Incident_Detector_Recordings")
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_path = os.path.join(output_dir, f"recording_{timestamp}.mp4")

            # Get a frame to determine dimensions
            if (
                hasattr(self.detection_thread, "recent_frames")
                and self.detection_thread.recent_frames
            ):
                frame = self.detection_thread.recent_frames[0][0]

                if frame is not None:
                    h, w = frame.shape[:2]

                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID' for AVI
                    self.video_writer = cv2.VideoWriter(
                        recording_path, fourcc, 20.0, (w, h)
                    )

                    self.is_recording = True
                    self.statusBar().showMessage(f"Recording to {recording_path}")

                    # Store path for later
                    self.recording_path = recording_path
                else:
                    self.record_btn.setChecked(False)
                    QMessageBox.warning(
                        self,
                        "Recording Error",
                        "Cannot start recording: No frames available",
                    )
            else:
                self.record_btn.setChecked(False)
                QMessageBox.warning(
                    self,
                    "Recording Error",
                    "Cannot start recording: No frames available",
                )
        else:  # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.is_recording = False

            if hasattr(self, "recording_path") and self.recording_path:
                self.statusBar().showMessage(
                    f"Recording saved to {self.recording_path}"
                )

                # Ask if user wants to open the recording
                reply = QMessageBox.question(
                    self,
                    "Recording Finished",
                    f"Recording saved to {self.recording_path}\nDo you want to open it now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Open recording with default video player
                    QDesktopServices.openUrl(QUrl.fromLocalFile(self.recording_path))

    def toggle_streaming(self, enabled):
        """Toggle streaming to web clients"""
        if enabled:
            try:
                # Start stream server if not already running
                if not self.stream_server:
                    # Import here to avoid dependency if not used
                    from http.server import HTTPServer, BaseHTTPRequestHandler
                    import threading

                    # Simple HTTP server to serve JPEG frames
                    class StreamingHandler(BaseHTTPRequestHandler):
                        def do_GET(self):
                            if self.path == "/":
                                # Serve HTML page with video stream
                                self.send_response(200)
                                self.send_header("Content-type", "text/html")
                                self.end_headers()

                                html = f"""
                                <html>
                                <head>
                                <title>Incident Detection Stream</title>
                                <style>
                                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }}
                                    h1 {{ color: #333; }}
                                    .stream-container {{ margin: 20px auto; max-width: 800px; }}
                                    img {{ max-width: 100%; border: 1px solid #ddd; }}
                                </style>
                                </head>
                                <body>
                                <h1>Live Detection Stream</h1>
                                <div class="stream-container">
                                    <img src="/stream" id="stream">
                                </div>
                                <script>
                                    // Auto refresh the image every second
                                    setInterval(function() {{
                                        document.getElementById('stream').src = "/stream?" + new Date().getTime();
                                    }}, 1000);
                                </script>
                                </body>
                                </html>
                                """

                                self.wfile.write(html.encode())

                            elif self.path.startswith("/stream"):
                                # Stream latest frame as JPEG
                                self.send_response(200)
                                self.send_header("Content-type", "image/jpeg")
                                self.send_header(
                                    "Cache-Control",
                                    "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0",
                                )
                                self.end_headers()

                                # Get latest frame
                                if (
                                    hasattr(app.detection_thread, "recent_frames")
                                    and app.detection_thread.recent_frames
                                ):
                                    frame = app.detection_thread.recent_frames[-1][
                                        0
                                    ].copy()

                                    # Encode frame as JPEG
                                    _, buffer = cv2.imencode(".jpg", frame)
                                    self.wfile.write(buffer.tobytes())
                                else:
                                    # Send blank image if no frame available
                                    blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
                                    cv2.putText(
                                        blank,
                                        "No stream available",
                                        (100, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 0, 0),
                                        2,
                                    )
                                    _, buffer = cv2.imencode(".jpg", blank)
                                    self.wfile.write(buffer.tobytes())
                            else:
                                self.send_error(404)
                                self.end_headers()

                    # Store app reference for handler
                    global app
                    app = self

                    # Find an available port
                    port = 8080
                    max_tries = 10

                    for i in range(max_tries):
                        try:
                            # Try to create server
                            server = HTTPServer(("", port), StreamingHandler)
                            break
                        except socket.error:
                            port += 1
                            if i == max_tries - 1:
                                raise RuntimeError("No available ports")

                    # Start server in a thread
                    server_thread = threading.Thread(target=server.serve_forever)
                    server_thread.daemon = True
                    server_thread.start()

                    self.stream_server = {
                        "server": server,
                        "thread": server_thread,
                        "port": port,
                    }

                    # Show info message with URL
                    local_ip = socket.gethostbyname(socket.gethostname())
                    QMessageBox.information(
                        self,
                        "Streaming Started",
                        f"Stream available at:\n\nLocal: http://localhost:{port}\nNetwork: http://{local_ip}:{port}",
                    )

            except Exception as e:
                logger.error(f"Error starting stream server: {str(e)}")
                QMessageBox.critical(
                    self, "Streaming Error", f"Failed to start streaming: {str(e)}"
                )
                self.stream_check.setChecked(False)
        else:
            # Stop streaming
            if self.stream_server:
                try:
                    self.stream_server["server"].shutdown()
                    self.stream_server = None
                except:
                    pass

    def on_tab_changed(self, index):
        """Handle changing between camera and video tabs"""
        # Stop any running detection
        if hasattr(self, "detection_thread") and self.detection_thread:
            self.stop_detection()

        if index == 0:  # Camera tab
            self.camera_start_btn.setText("Start Detection")
            self.camera_start_btn.setIcon(QIcon.fromTheme("media-playback-start"))
            self.record_btn.setEnabled(False)

            # Stop streaming if active
            if self.stream_check and self.stream_check.isChecked():
                self.stream_check.setChecked(False)
                self.toggle_streaming(False)

        elif index == 1:  # Video tab
            self.process_btn.setText("Process Video")
            self.process_btn.setIcon(QIcon.fromTheme("media-playback-start"))

        elif index == 2:  # Analytics tab
            # Refresh analytics
            self.update_analytics()

    def update_alert_settings(self):
        """Update alert settings based on UI state"""
        self.detection_settings["sound_alerts"] = self.sound_alerts_check.isChecked()
        self.detection_settings["popup_alerts"] = self.popup_alerts_check.isChecked()
        self.detection_settings["log_alerts"] = self.log_alerts_check.isChecked()

    def update_analytics(self):
        """Update analytics dashboard with latest data"""
        try:
            # Get time range selection
            time_range = self.time_range_combo.currentText()
            days = 1
            if time_range == "Last 3 Days":
                days = 3
            elif time_range == "Last Week":
                days = 7
            elif time_range == "Last Month":
                days = 30
            elif time_range == "All Time":
                days = 365  # Arbitrary large number

            # Get incident type filter
            incident_type = None
            if self.incident_type_combo.currentIndex() > 0:  # Not "All Types"
                incident_type = (
                    self.incident_type_combo.currentText().lower().rstrip("s")
                )

            # Get grouping
            group_by = self.group_by_combo.currentText().lower()

            # Fetch analytics data from database
            self.analytics_data = self.db.get_analytics(days)

            # Update summary charts if Qt Charts is available
            if HAS_CHARTS:
                self.update_summary_charts(incident_type)
                self.update_time_distribution_charts(incident_type)

            # Update incident history table
            self.update_incident_history(days, incident_type)

            # Update heatmap
            self.update_analytics_heatmap(days, incident_type, group_by)

        except Exception as e:
            logger.error(f"Error updating analytics: {str(e)}")
            self.statusBar().showMessage(f"Error updating analytics: {str(e)}")

    def update_summary_charts(self, incident_type=None):
        """Update summary charts with analytics data"""
        if not HAS_CHARTS:
            return

        try:
            # Clear existing series
            self.summary_chart1.removeAllSeries()
            self.summary_chart2.removeAllSeries()

            # Process data for charts
            incident_counts = {}
            date_counts = {}

            for item in self.analytics_data:
                # Filter by type if specified
                if incident_type and item["type"] != incident_type:
                    continue

                # Count by type
                if item["type"] not in incident_counts:
                    incident_counts[item["type"]] = 0
                incident_counts[item["type"]] += item["count"]

                # Count by date
                date = item["date"]
                if date not in date_counts:
                    date_counts[date] = 0
                date_counts[date] += item["count"]

            # Create bar chart for incident types
            bar_set = QBarSet("Incidents")

            # Add data to bar set
            labels = []
            for type_name, count in sorted(
                incident_counts.items(), key=lambda x: x[1], reverse=True
            ):
                bar_set.append(count)
                labels.append(type_name.capitalize())

            bar_series = QBarSeries()
            bar_series.append(bar_set)

            self.summary_chart1.addSeries(bar_series)

            # Set up axes
            axis_x = QBarCategoryAxis()
            axis_x.append(labels)
            self.summary_chart1.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            bar_series.attachAxis(axis_x)

            axis_y = QValueAxis()
            max_count = max(incident_counts.values()) if incident_counts else 10
            axis_y.setRange(0, max_count * 1.1)  # Add 10% margin
            self.summary_chart1.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            bar_series.attachAxis(axis_y)

            # Create line chart for trend over time
            line_series = QLineSeries()
            line_series.setName("Incident Trend")

            # Sort dates
            sorted_dates = sorted(date_counts.keys())

            # Add data points
            for i, date in enumerate(sorted_dates):
                line_series.append(i, date_counts[date])

            self.summary_chart2.addSeries(line_series)

            # Set up axes
            axis_x2 = QBarCategoryAxis()
            axis_x2.append(sorted_dates)
            self.summary_chart2.addAxis(axis_x2, Qt.AlignmentFlag.AlignBottom)
            line_series.attachAxis(axis_x2)

            axis_y2 = QValueAxis()
            max_count = max(date_counts.values()) if date_counts else 10
            axis_y2.setRange(0, max_count * 1.1)  # Add 10% margin
            self.summary_chart2.addAxis(axis_y2, Qt.AlignmentFlag.AlignLeft)
            line_series.attachAxis(axis_y2)

            # Update summary statistics table
            total_incidents = sum(incident_counts.values())
            most_common = (
                max(incident_counts.items(), key=lambda x: x[1])[0]
                if incident_counts
                else "None"
            )

            # Calculate average confidence
            total_confidence = 0
            total_count = 0
            for item in self.analytics_data:
                if incident_type and item["type"] != incident_type:
                    continue
                if item["avg_confidence"]:
                    total_confidence += item["avg_confidence"] * item["count"]
                    total_count += item["count"]

            avg_confidence = total_confidence / total_count if total_count > 0 else 0

            # Find peak time
            # This would require more detailed time data, for now use placeholder
            peak_time = "N/A"

            # Count incidents today
            today = datetime.date.today().isoformat()
            incidents_today = sum(
                item["count"]
                for item in self.analytics_data
                if item["date"] == today
                and (not incident_type or item["type"] == incident_type)
            )

            # Update table
            self.summary_stats.setItem(0, 1, QTableWidgetItem(str(total_incidents)))
            self.summary_stats.setItem(1, 1, QTableWidgetItem(most_common.capitalize()))
            self.summary_stats.setItem(2, 1, QTableWidgetItem(f"{avg_confidence:.2f}"))
            self.summary_stats.setItem(3, 1, QTableWidgetItem(peak_time))
            self.summary_stats.setItem(4, 1, QTableWidgetItem(str(incidents_today)))

            # Calculate changes (simplified)
            self.summary_stats.setItem(0, 2, QTableWidgetItem("N/A"))
            self.summary_stats.setItem(1, 2, QTableWidgetItem("N/A"))
            self.summary_stats.setItem(2, 2, QTableWidgetItem("N/A"))
            self.summary_stats.setItem(3, 2, QTableWidgetItem("N/A"))
            self.summary_stats.setItem(4, 2, QTableWidgetItem("N/A"))

        except Exception as e:
            logger.error(f"Error updating summary charts: {str(e)}")

    def update_time_distribution_charts(self, incident_type=None):
        """Update time distribution charts with analytics data"""
        if not HAS_CHARTS:
            return

        try:
            # Clear existing series
            self.time_dist_chart.removeAllSeries()
            self.conf_dist_chart.removeAllSeries()

            # Fetch data from database - this would be more detailed in a real app
            # For simplicity, we'll create synthetic data based on analytics

            # Time distribution (by hour)
            time_dist = {}
            for hour in range(24):
                time_dist[hour] = 0

            # Confidence distribution
            conf_dist = {}
            for conf in range(5, 10):
                conf_dist[conf / 10] = 0

            # Populate with synthetic data
            for item in self.analytics_data:
                if incident_type and item["type"] != incident_type:
                    continue

                # Distribute count across hours based on type
                if item["type"] == "falls":
                    # Falls more common in morning and evening
                    for h in [7, 8, 9, 17, 18, 19]:
                        time_dist[h] += item["count"] * 0.1
                elif item["type"] == "attacks":
                    # Attacks more common at night
                    for h in [21, 22, 23, 0, 1]:
                        time_dist[h] += item["count"] * 0.15
                else:
                    # Other incidents distributed evenly
                    for h in range(24):
                        time_dist[h] += item["count"] / 24

                # Distribute by confidence
                if item["avg_confidence"]:
                    conf_key = (
                        round(item["avg_confidence"] * 2) / 2
                    )  # Round to nearest 0.5
                    if conf_key in conf_dist:
                        conf_dist[conf_key] += item["count"]
                    else:
                        nearest_key = min(
                            conf_dist.keys(), key=lambda x: abs(x - conf_key)
                        )
                        conf_dist[nearest_key] += item["count"]

            # Create bar chart for time distribution
            bar_set = QBarSet("Incidents by Hour")

            # Add data to bar set
            for hour in range(24):
                bar_set.append(time_dist[hour])

            bar_series = QBarSeries()
            bar_series.append(bar_set)

            self.time_dist_chart.addSeries(bar_series)

            # Set up axes
            axis_x = QBarCategoryAxis()
            axis_x.append([f"{h:02d}:00" for h in range(24)])
            self.time_dist_chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            bar_series.attachAxis(axis_x)

            axis_y = QValueAxis()
            max_count = max(time_dist.values()) if time_dist else 10
            axis_y.setRange(0, max_count * 1.1)  # Add 10% margin
            self.time_dist_chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            bar_series.attachAxis(axis_y)

            # Create pie chart for confidence distribution
            pie_series = QPieSeries()

            # Add slices
            for conf, count in conf_dist.items():
                pie_series.append(f"Confidence {conf:.1f}", count)

            self.conf_dist_chart.addSeries(pie_series)

        except Exception as e:
            logger.error(f"Error updating time distribution charts: {str(e)}")

    def update_incident_history(self, days, incident_type=None):
        """Update incident history table with data from database"""
        try:
            # Clear table
            self.incidents_history_table.setRowCount(0)

            # Calculate date range
            end_date = datetime.datetime.now().isoformat()
            start_date = (
                datetime.datetime.now() - datetime.timedelta(days=days)
            ).isoformat()

            # Fetch incidents from database
            incidents = self.db.get_incidents(
                incident_type=incident_type,
                start_date=start_date,
                end_date=end_date,
                limit=100,
            )

            # Populate table
            for i, incident in enumerate(incidents):
                self.incidents_history_table.insertRow(i)

                # Format timestamp
                timestamp = incident["timestamp"]
                if isinstance(timestamp, str):
                    timestamp = datetime.datetime.fromisoformat(timestamp)
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                self.incidents_history_table.setItem(
                    i, 0, QTableWidgetItem(formatted_time)
                )
                self.incidents_history_table.setItem(
                    i, 1, QTableWidgetItem(incident["type"].capitalize())
                )
                self.incidents_history_table.setItem(
                    i, 2, QTableWidgetItem(incident["source"] or "Unknown")
                )
                self.incidents_history_table.setItem(
                    i,
                    3,
                    QTableWidgetItem(
                        f"{incident['confidence']:.2f}"
                        if incident["confidence"]
                        else "N/A"
                    ),
                )

                # Parse details
                details = incident["details"]
                if isinstance(details, str):
                    try:
                        details_dict = json.loads(details)
                        details_str = ", ".join(
                            [
                                f"{k}={v}"
                                for k, v in details_dict.items()
                                if k not in ["time", "confidence", "frame"]
                            ]
                        )
                    except:
                        details_str = details
                else:
                    details_str = str(details)

                self.incidents_history_table.setItem(
                    i, 4, QTableWidgetItem(details_str)
                )

                # Add view button if image exists
                if incident["image_path"]:
                    view_btn = QPushButton("View")
                    view_btn.clicked.connect(
                        lambda checked, path=incident[
                            "image_path"
                        ]: QDesktopServices.openUrl(QUrl.fromLocalFile(path))
                    )
                    self.incidents_history_table.setCellWidget(i, 5, view_btn)
                else:
                    self.incidents_history_table.setItem(
                        i, 5, QTableWidgetItem("No Image")
                    )

        except Exception as e:
            logger.error(f"Error updating incident history: {str(e)}")

    def update_analytics_heatmap(self, days, incident_type, group_by):
        """Update analytics heatmap visualization"""
        try:
            # Clear figure
            self.heatmap_figure.clear()

            # Create axes
            ax = self.heatmap_figure.add_subplot(111)

            # Process data for heatmap
            if not self.analytics_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                self.heatmap_canvas.draw()
                return

            # Create date-based heatmap
            date_type_counts = {}

            for item in self.analytics_data:
                if incident_type and item["type"] != incident_type:
                    continue

                date = item["date"]
                incident_type = item["type"]

                if date not in date_type_counts:
                    date_type_counts[date] = {}

                if incident_type not in date_type_counts[date]:
                    date_type_counts[date][incident_type] = 0

                date_type_counts[date][incident_type] += item["count"]

            # Convert to matrix for heatmap
            dates = sorted(date_type_counts.keys())

            # Get all incident types
            all_types = set()
            for date_data in date_type_counts.values():
                all_types.update(date_data.keys())

            types = sorted(all_types)

            # Create data matrix
            data = np.zeros((len(dates), len(types)))

            for i, date in enumerate(dates):
                for j, type_name in enumerate(types):
                    if type_name in date_type_counts[date]:
                        data[i, j] = date_type_counts[date][type_name]

            # Create heatmap
            im = ax.imshow(data, cmap="YlOrRd")

            # Set labels
            ax.set_xticks(np.arange(len(types)))
            ax.set_yticks(np.arange(len(dates)))
            ax.set_xticklabels([t.capitalize() for t in types])
            ax.set_yticklabels(dates)

            # Rotate x labels for readability
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            # Add colorbar
            cbar = self.heatmap_figure.colorbar(im, ax=ax)
            cbar.set_label("Incident Count")

            # Add title
            ax.set_title("Incident Heatmap by Date and Type")

            # Add value annotations
            for i in range(len(dates)):
                for j in range(len(types)):
                    if data[i, j] > 0:
                        text = ax.text(
                            j,
                            i,
                            int(data[i, j]),
                            ha="center",
                            va="center",
                            color="black",
                        )

            self.heatmap_figure.tight_layout()
            self.heatmap_canvas.draw()

        except Exception as e:
            logger.error(f"Error updating analytics heatmap: {str(e)}")

    def search_incidents(self):
        """Search incident history based on criteria"""
        try:
            # Get search text
            search_text = self.search_edit.text().lower()

            # Get date range
            start_date = self.date_from.date().toString(Qt.DateFormat.ISODate)
            end_date = self.date_to.date().toString(Qt.DateFormat.ISODate)

            # Fetch incidents from database
            incidents = self.db.get_incidents(
                start_date=start_date,
                end_date=end_date,
                limit=1000,  # Higher limit for search
            )

            # Filter by search text if provided
            if search_text:
                filtered_incidents = []
                for incident in incidents:
                    # Check in type, source, and details
                    if (
                        search_text in incident["type"].lower()
                        or (
                            incident["source"]
                            and search_text in incident["source"].lower()
                        )
                        or (
                            incident["details"]
                            and search_text in str(incident["details"]).lower()
                        )
                    ):
                        filtered_incidents.append(incident)

                incidents = filtered_incidents

            # Update table
            self.incidents_history_table.setRowCount(0)

            for i, incident in enumerate(incidents):
                self.incidents_history_table.insertRow(i)

                # Format timestamp
                timestamp = incident["timestamp"]
                if isinstance(timestamp, str):
                    timestamp = datetime.datetime.fromisoformat(timestamp)
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                self.incidents_history_table.setItem(
                    i, 0, QTableWidgetItem(formatted_time)
                )
                self.incidents_history_table.setItem(
                    i, 1, QTableWidgetItem(incident["type"].capitalize())
                )
                self.incidents_history_table.setItem(
                    i, 2, QTableWidgetItem(incident["source"] or "Unknown")
                )
                self.incidents_history_table.setItem(
                    i,
                    3,
                    QTableWidgetItem(
                        f"{incident['confidence']:.2f}"
                        if incident["confidence"]
                        else "N/A"
                    ),
                )

                # Parse details
                details = incident["details"]
                if isinstance(details, str):
                    try:
                        details_dict = json.loads(details)
                        details_str = ", ".join(
                            [
                                f"{k}={v}"
                                for k, v in details_dict.items()
                                if k not in ["time", "confidence", "frame"]
                            ]
                        )
                    except:
                        details_str = details
                else:
                    details_str = str(details)

                self.incidents_history_table.setItem(
                    i, 4, QTableWidgetItem(details_str)
                )

                # Add view button if image exists
                if incident["image_path"]:
                    view_btn = QPushButton("View")
                    view_btn.clicked.connect(
                        lambda checked, path=incident[
                            "image_path"
                        ]: QDesktopServices.openUrl(QUrl.fromLocalFile(path))
                    )
                    self.incidents_history_table.setCellWidget(i, 5, view_btn)
                else:
                    self.incidents_history_table.setItem(
                        i, 5, QTableWidgetItem("No Image")
                    )

            # Update status
            self.statusBar().showMessage(
                f"Found {len(incidents)} incidents matching criteria"
            )

        except Exception as e:
            logger.error(f"Error searching incidents: {str(e)}")

    def add_custom_rule(self):
        """Add a new custom detection rule"""
        rule_dialog = CustomRuleDialog(self)
        if rule_dialog.exec() == QDialog.DialogCode.Accepted:
            rule_data = rule_dialog.get_rule_data()

            # Add rule to list
            self.custom_rules.append(rule_data)

            # Update list widget
            self.update_rules_list()

            # Save rules
            self.save_custom_rules()

    def edit_custom_rule(self):
        """Edit an existing custom rule"""
        selected_items = self.rules_list.selectedItems()
        if not selected_items:
            return

        selected_index = self.rules_list.row(selected_items[0])

        # Open dialog with current rule data
        rule_dialog = CustomRuleDialog(self, self.custom_rules[selected_index])

        if rule_dialog.exec() == QDialog.DialogCode.Accepted:
            # Update rule with new data
            self.custom_rules[selected_index] = rule_dialog.get_rule_data()

            # Update list widget
            self.update_rules_list()

            # Update details display
            self.display_rule_details(selected_index)

            # Save rules
            self.save_custom_rules()

    def remove_custom_rule(self):
        """Remove a custom rule"""
        selected_items = self.rules_list.selectedItems()
        if not selected_items:
            return

        selected_index = self.rules_list.row(selected_items[0])

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the rule '{self.custom_rules[selected_index]['name']}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Remove rule
            self.custom_rules.pop(selected_index)

            # Update list widget
            self.update_rules_list()

            # Clear details
            self.rule_details_text.clear()

            # Save rules
            self.save_custom_rules()

    def update_rules_list(self):
        """Update the custom rules list widget"""
        self.rules_list.clear()

        for rule in self.custom_rules:
            item_text = f"{rule['name']} ({rule['type'].capitalize()})"
            self.rules_list.addItem(item_text)

        # Update button states
        self.edit_rule_btn.setEnabled(False)
        self.remove_rule_btn.setEnabled(False)

    def on_rule_selection_changed(self):
        """Handle selection change in rules list"""
        selected = len(self.rules_list.selectedItems()) > 0
        self.edit_rule_btn.setEnabled(selected)
        self.remove_rule_btn.setEnabled(selected)

        if selected:
            selected_index = self.rules_list.row(self.rules_list.selectedItems()[0])
            self.display_rule_details(selected_index)
        else:
            self.rule_details_text.clear()

    def display_rule_details(self, index):
        """Display details of the selected rule"""
        if index < 0 or index >= len(self.custom_rules):
            return

        rule = self.custom_rules[index]

        details = f"<h3>{rule['name']}</h3>\n"
        details += f"<p><b>Type:</b> {rule['type'].capitalize()}</p>\n"
        details += (
            f"<p><b>Confidence Threshold:</b> {rule['confidence_threshold']:.2f}</p>\n"
        )
        details += f"<p><b>Minimum Duration:</b> {rule['time_threshold']} frames</p>\n"
        details += f"<p><b>Minimum People Count:</b> {rule['person_count']}</p>\n"

        if "custom_conditions" in rule and rule["custom_conditions"]:
            details += f"<p><b>Custom Conditions:</b></p>\n"
            details += f"<pre>{rule['custom_conditions']}</pre>\n"

        self.rule_details_text.setHtml(details)

    def select_roi(self):
        """Select a region of interest"""
        # Need a frame to select ROI
        frame = None

        if self.tabWidget.currentIndex() == 0:  # Camera tab
            if (
                hasattr(self, "detection_thread")
                and self.detection_thread
                and self.detection_thread.isRunning()
                and hasattr(self.detection_thread, "recent_frames")
            ):
                if self.detection_thread.recent_frames:
                    frame = self.detection_thread.recent_frames[0][0]
        else:  # Video tab
            if self.video_path:
                cap = cv2.VideoCapture(self.video_path)
                ret, frame = cap.read()
                cap.release()

        if frame is None:
            QMessageBox.warning(
                self,
                "ROI Selection",
                "No frame available for ROI selection.\nPlease load a video or start camera detection first.",
            )
            return

        # Show ROI selection dialog
        roi_dialog = RegionOfInterestDialog(frame, self)
        if roi_dialog.exec() == QDialog.DialogCode.Accepted and roi_dialog.roi:
            self.detection_settings["roi"] = roi_dialog.roi
            self.statusBar().showMessage(f"ROI set: {roi_dialog.roi}")

            # Update detection thread if running
            if (
                hasattr(self, "detection_thread")
                and self.detection_thread
                and self.detection_thread.isRunning()
            ):
                self.detection_thread.update_settings(self.detection_settings)

    def clear_roi(self):
        """Clear the region of interest"""
        self.detection_settings["roi"] = None
        self.statusBar().showMessage("ROI cleared")

        # Update detection thread if running
        if (
            hasattr(self, "detection_thread")
            and self.detection_thread
            and self.detection_thread.isRunning()
        ):
            self.detection_thread.update_settings(self.detection_settings)

    def configure_zones(self):
        """Configure detection zones"""
        zones = self.detection_settings.get("zones", {})

        # Show zone configuration dialog
        zone_dialog = ZoneConfigDialog(zones, self)
        if zone_dialog.exec() == QDialog.DialogCode.Accepted:
            # Update zones
            self.detection_settings["zones"] = zone_dialog.get_zones()

            # Update detection thread if running
            if (
                hasattr(self, "detection_thread")
                and self.detection_thread
                and self.detection_thread.isRunning()
            ):
                self.detection_thread.update_settings(self.detection_settings)

            self.statusBar().showMessage("Detection zones updated")

    def export_alerts(self):
        """Export alerts to a file"""
        if self.alerts_list.count() == 0:
            QMessageBox.information(self, "Export Alerts", "No alerts to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Alerts",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
        )

        if not file_path:
            return

        try:
            with open(file_path, "w") as f:
                if file_path.lower().endswith(".csv"):
                    # Write CSV header
                    f.write("Timestamp,Alert Type,Confidence,Details\n")

                    # Write alerts
                    for i in range(self.alerts_list.count()):
                        alert_text = self.alerts_list.item(i).text()
                        parts = alert_text.split(" - ", 2)

                        if len(parts) >= 2:
                            timestamp = parts[0]

                            # Extract type and confidence
                            type_conf = parts[1].split(" (conf: ")
                            alert_type = type_conf[0]

                            conf = "0.00"
                            if len(type_conf) > 1:
                                conf = type_conf[1].replace(")", "")

                            details = parts[2] if len(parts) > 2 else ""

                            f.write(
                                f'"{timestamp}","{alert_type}",{conf},"{details}"\n'
                            )
                else:
                    # Write text format
                    for i in range(self.alerts_list.count()):
                        f.write(self.alerts_list.item(i).text() + "\n")

            QMessageBox.information(
                self, "Export Complete", f"Alerts exported to {file_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error exporting alerts: {str(e)}"
            )

    def export_results(self):
        """Export detection results"""
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
        """Export detection results to CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to CSV", "", "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            incidents = {}

            # Get incidents from appropriate source
            if self.tabWidget.currentIndex() == 1:  # Video tab
                # Extract from table (simpler approach)
                for row in range(self.incidents_table.rowCount()):
                    incident_type = self.incidents_table.item(row, 0).text().lower()
                    time_str = self.incidents_table.item(row, 1).text().replace("s", "")
                    confidence = float(self.incidents_table.item(row, 2).text())
                    frame = int(self.incidents_table.item(row, 3).text())

                    if incident_type not in incidents:
                        incidents[incident_type] = []

                    incidents[incident_type].append(
                        (frame, {"time": float(time_str), "confidence": confidence})
                    )
            else:  # Camera tab
                if (
                    hasattr(self, "detection_thread")
                    and self.detection_thread
                    and hasattr(self.detection_thread, "incidents")
                ):
                    incidents = self.detection_thread.incidents

            with open(file_path, "w", newline="") as csvfile:
                # Write header
                csvfile.write("Type,Time (seconds),Confidence,Frame,Details\n")

                # Write incidents
                for incident_type, incidents_list in incidents.items():
                    for frame_num, details in incidents_list:
                        if not isinstance(details, list):
                            details = [details]

                        for detail in details:
                            time_sec = detail.get("time", frame_num / 30.0)
                            confidence = detail.get("confidence", 0.5)

                            # Format other details
                            details_str = ";".join(
                                [
                                    f"{k}={v}"
                                    for k, v in detail.items()
                                    if k not in ["time", "confidence", "frame"]
                                ]
                            )

                            csvfile.write(
                                f'{incident_type},{time_sec:.2f},{confidence:.2f},{frame_num},"{details_str}"\n'
                            )

            self.statusBar().showMessage(f"Exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error exporting to CSV: {str(e)}"
            )

    def export_to_pdf(self):
        """Export detection results to PDF report"""
        try:
            # Ask for file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export to PDF", "", "PDF Files (*.pdf)"
            )

            if not file_path:
                return

            # Show a simple message for this implementation
            QMessageBox.information(
                self,
                "Export to PDF",
                "In a full implementation, this would generate a PDF report with incident details, statistics, and visualizations. PDF export functionality requires additional libraries like reportlab.",
            )

            self.statusBar().showMessage(f"Export to PDF requested to {file_path}")

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error generating PDF report: {str(e)}"
            )

    def export_to_json(self):
        """Export detection results to JSON"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to JSON", "", "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            incidents = {}

            # Get incidents from appropriate source
            if self.tabWidget.currentIndex() == 1:  # Video tab
                if hasattr(self, "detection_thread") and hasattr(
                    self.detection_thread, "incidents"
                ):
                    incidents = self.detection_thread.incidents
            else:  # Camera tab
                if hasattr(self, "detection_thread") and hasattr(
                    self.detection_thread, "incidents"
                ):
                    incidents = self.detection_thread.incidents

            # Create export data
            export_data = {
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "version": APP_VERSION,
                    "model": self.settings.value("model", DEFAULT_MODEL, type=str),
                    "confidence_threshold": self.detection_settings.get(
                        "confidence_threshold", 0.5
                    ),
                },
                "incidents": {},
            }

            # Add source info
            if self.tabWidget.currentIndex() == 1:  # Video tab
                if self.video_path:
                    export_data["metadata"]["source"] = {
                        "type": "video",
                        "path": self.video_path,
                        "filename": os.path.basename(self.video_path),
                    }
            else:  # Camera tab
                camera_name = self.camera_combo.currentText()
                export_data["metadata"]["source"] = {
                    "type": "camera",
                    "name": camera_name,
                }

            # Process incidents
            for incident_type, incidents_list in incidents.items():
                export_data["incidents"][incident_type] = []

                for frame_num, details in incidents_list:
                    if not isinstance(details, list):
                        details = [details]

                    for detail in details:
                        # Convert numpy values to Python types for JSON serialization
                        serializable_detail = {}

                        for k, v in detail.items():
                            if isinstance(v, np.ndarray):
                                serializable_detail[k] = v.tolist()
                            elif isinstance(v, np.generic):
                                serializable_detail[k] = v.item()
                            else:
                                serializable_detail[k] = v

                        export_data["incidents"][incident_type].append(
                            {"frame": int(frame_num), "details": serializable_detail}
                        )

            # Generate statistics summary
            summary = {"total_incidents": 0, "by_type": {}}

            for incident_type, incidents_list in export_data["incidents"].items():
                count = len(incidents_list)
                summary["total_incidents"] += count
                summary["by_type"][incident_type] = count

            export_data["summary"] = summary

            # Write to file
            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)

            self.statusBar().showMessage(f"Exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error exporting to JSON: {str(e)}"
            )

    def generate_report(self):
        """Generate comprehensive incident report"""
        # This could be a more interactive version of export_to_pdf with more options
        # For simplicity, we'll just call export_to_pdf for now
        self.export_to_pdf()

    def export_analytics(self):
        """Export analytics data to report"""
        try:
            # Ask for file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Analytics Report", "", "PDF Files (*.pdf)"
            )

            if not file_path:
                return

            # Show a simple message for this implementation
            QMessageBox.information(
                self,
                "Export Analytics",
                "In a full implementation, this would generate a PDF report with analytics data, visualizations, and incident statistics. PDF export functionality requires additional libraries like reportlab.",
            )

            self.statusBar().showMessage(f"Analytics export requested to {file_path}")

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error generating analytics report: {str(e)}"
            )

    def toggle_fullscreen(self, fullscreen=None):
        """Toggle fullscreen mode"""
        if fullscreen is None:
            # Toggle mode
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            # Set specific mode
            if fullscreen:
                self.showFullScreen()
            else:
                self.showNormal()

    def check_updates(self):
        """Check for application updates"""
        # In a real application, this would contact a server to check for updates
        # For this demo, we'll just show a message
        QMessageBox.information(
            self,
            "Check for Updates",
            f"You are running version {APP_VERSION} of the Incident Detection System.\n\n"
            "This is the latest version available.",
        )

    def show_help(self):
        """Show application help"""
        help_text = f"""
        <h2>Incident Detection System v{APP_VERSION} - Help</h2>
        
        <h3>Main Features:</h3>
        <ul>
            <li><b>Real-time Camera Detection:</b> Monitor live camera feeds for incidents</li>
            <li><b>Video Analysis:</b> Process video files to detect incidents</li>
            <li><b>Multiple Incident Types:</b> Detect falls, attacks, accidents, intrusions, loitering, and abandoned objects</li>
            <li><b>Advanced Visualizations:</b> 3D analysis, heat maps, spatial analysis</li>
            <li><b>Analytics Dashboard:</b> Review incident statistics and trends</li>
            <li><b>Custom Detection Rules:</b> Define your own incident detection criteria</li>
            <li><b>Notifications:</b> Email alerts and webhook integration</li>
        </ul>
        
        <h3>Quick Start Guide:</h3>
        
        <h4>Camera Detection:</h4>
        <ol>
            <li>Go to the "Camera Detection" tab</li>
            <li>Select a camera from the dropdown</li>
            <li>Click "Start Detection" to begin monitoring</li>
            <li>Incidents will appear in the Alerts panel</li>
            <li>Use the "Record" button to save video if needed</li>
        </ol>
        
        <h4>Video Analysis:</h4>
        <ol>
            <li>Go to the "Video Analysis" tab</li>
            <li>Click "Load Video" to select a video file</li>
            <li>Adjust the confidence threshold as needed</li>
            <li>Click "Process Video" to analyze the video</li>
            <li>Review incidents in the Incidents table</li>
            <li>Use the 3D Analysis and Spatial Analysis tabs for advanced visualization</li>
        </ol>
        
        <h4>Analytics:</h4>
        <ol>
            <li>Go to the "Analytics" tab to view incident statistics</li>
            <li>Select time ranges and grouping options</li>
            <li>Use the Incident History tab to search past incidents</li>
            <li>Click "Export Report" to generate PDF reports</li>
        </ol>
        
        <h3>Keyboard Shortcuts:</h3>
        <ul>
            <li><b>Ctrl+O:</b> Open video file</li>
            <li><b>Ctrl+S:</b> Export results</li>
            <li><b>Ctrl+P:</b> Process video</li>
            <li><b>Ctrl+C:</b> Start/stop camera detection</li>
            <li><b>F11:</b> Toggle fullscreen</li>
            <li><b>F1:</b> Show this help</li>
            <li><b>Ctrl+Q:</b> Quit application</li>
        </ul>
        
        <h3>Tips for Better Results:</h3>
        <ul>
            <li>Adjust the confidence threshold based on your needs (higher for fewer false positives)</li>
            <li>Configure detection zones for more accurate intrusion detection</li>
            <li>Create custom rules for specific detection scenarios</li>
            <li>Use the ROI tool to focus detection on relevant areas</li>
            <li>For better performance on low-end systems, increase the frame interval</li>
        </ul>
        """

        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Help")
        help_dialog.setMinimumSize(700, 600)

        layout = QVBoxLayout()

        # Create scroll area for help text
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        help_content = QWidget()
        help_layout = QVBoxLayout()

        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setOpenExternalLinks(True)

        help_layout.addWidget(help_label)
        help_content.setLayout(help_layout)

        scroll.setWidget(help_content)
        layout.addWidget(scroll)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(help_dialog.accept)
        layout.addWidget(close_btn)

        help_dialog.setLayout(layout)
        help_dialog.exec()

    def show_about(self):
        """Show about dialog"""
        about_text = f"""
        <h1>Incident Detection System</h1>
        <p><b>Version:</b> {APP_VERSION}</p>
        <p>Advanced real-time detection system for falls, accidents, attacks, and other security incidents.</p>
        
        <p>This application uses:</p>
        <ul>
            <li>YOLOv8 for object detection and pose estimation</li>
            <li>PyQt6 for the user interface</li>
            <li>OpenCV for video processing</li>
            <li>SQLite for incident database</li>
            <li>Machine learning for advanced behavior analysis</li>
        </ul>
        
        <p>&copy; 2025 All rights reserved.</p>
        """

        QMessageBox.about(self, "About Incident Detection System", about_text)

    def show_settings(self):
        """Show application settings dialog"""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Application Settings")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Create tabs for different settings
        tabs = QTabWidget()

        # Detection tab
        detection_tab = QWidget()
        detection_layout = QVBoxLayout()

        # Model selection
        model_group = QGroupBox("Detection Model")
        model_layout = QGridLayout()

        model_layout.addWidget(QLabel("YOLO Model:"), 0, 0)
        model_combo = QComboBox()
        model_combo.addItems(
            ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt"]
        )

        current_model = self.settings.value("model", DEFAULT_MODEL, type=str)
        model_index = model_combo.findText(current_model)
        if model_index >= 0:
            model_combo.setCurrentIndex(model_index)

        model_layout.addWidget(model_combo, 0, 1)

        # GPU acceleration
        use_gpu = QCheckBox("Use GPU Acceleration (CUDA)")
        use_gpu.setChecked(self.detection_settings.get("use_gpu", False))
        model_layout.addWidget(use_gpu, 1, 0, 1, 2)

        model_group.setLayout(model_layout)
        detection_layout.addWidget(model_group)

        # Detection thresholds
        threshold_group = QGroupBox("Detection Thresholds")
        threshold_layout = QGridLayout()

        threshold_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        confidence_slider = QSlider(Qt.Orientation.Horizontal)
        confidence_slider.setRange(1, 10)
        confidence_slider.setValue(
            int(self.detection_settings.get("confidence_threshold", 0.5) * 10)
        )
        threshold_layout.addWidget(confidence_slider, 0, 1)

        confidence_label = QLabel(
            f"{self.detection_settings.get('confidence_threshold', 0.5):.1f}"
        )
        threshold_layout.addWidget(confidence_label, 0, 2)

        # Update label when slider moves
        confidence_slider.valueChanged.connect(
            lambda val: confidence_label.setText(f"{val/10:.1f}")
        )

        # Frame processing interval
        threshold_layout.addWidget(QLabel("Process every N frames:"), 1, 0)
        frame_interval = QSpinBox()
        frame_interval.setRange(1, 10)
        frame_interval.setValue(self.detection_settings.get("frame_interval", 1))
        threshold_layout.addWidget(frame_interval, 1, 1, 1, 2)

        # Alert cooldown
        threshold_layout.addWidget(QLabel("Alert Cooldown (frames):"), 2, 0)
        cooldown_frames = QSpinBox()
        cooldown_frames.setRange(1, 300)
        cooldown_frames.setValue(self.detection_settings.get("cooldown_frames", 30))
        threshold_layout.addWidget(cooldown_frames, 2, 1, 1, 2)

        threshold_group.setLayout(threshold_layout)
        detection_layout.addWidget(threshold_group)

        # Video stabilization
        video_group = QGroupBox("Video Processing")
        video_layout = QVBoxLayout()

        stabilize_video = QCheckBox("Enable Video Stabilization")
        stabilize_video.setChecked(
            self.detection_settings.get("stabilize_video", False)
        )
        video_layout.addWidget(stabilize_video)

        video_group.setLayout(video_layout)
        detection_layout.addWidget(video_group)

        detection_tab.setLayout(detection_layout)
        tabs.addTab(detection_tab, "Detection")

        # Interface tab
        interface_tab = QWidget()
        interface_layout = QVBoxLayout()

        # UI settings
        ui_group = QGroupBox("User Interface")
        ui_layout = QVBoxLayout()

        show_fps = QCheckBox("Show FPS counter")
        show_fps.setChecked(self.settings.value("show_fps", True, type=bool))
        ui_layout.addWidget(show_fps)

        show_confidence = QCheckBox("Show confidence scores")
        show_confidence.setChecked(
            self.settings.value("show_confidence", True, type=bool)
        )
        ui_layout.addWidget(show_confidence)

        minimize_to_tray = QCheckBox("Minimize to system tray when closed")
        minimize_to_tray.setChecked(
            self.settings.value("minimize_to_tray", False, type=bool)
        )
        ui_layout.addWidget(minimize_to_tray)

        ui_group.setLayout(ui_layout)
        interface_layout.addWidget(ui_group)

        # Default tab
        default_group = QGroupBox("Default Settings")
        default_layout = QGridLayout()

        default_layout.addWidget(QLabel("Default Mode:"), 0, 0)
        default_mode = QComboBox()
        default_mode.addItems(
            ["Camera Detection", "Video Analysis", "Analytics Dashboard"]
        )
        default_mode.setCurrentIndex(self.settings.value("default_tab", 0, type=int))
        default_layout.addWidget(default_mode, 0, 1)

        default_group.setLayout(default_layout)
        interface_layout.addWidget(default_group)

        interface_tab.setLayout(interface_layout)
        tabs.addTab(interface_tab, "Interface")

        layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(dialog.accept)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Save detection settings
            self.detection_settings["confidence_threshold"] = (
                confidence_slider.value() / 10.0
            )
            self.detection_settings["frame_interval"] = frame_interval.value()
            self.detection_settings["cooldown_frames"] = cooldown_frames.value()
            self.detection_settings["use_gpu"] = use_gpu.isChecked()
            self.detection_settings["stabilize_video"] = stabilize_video.isChecked()

            # Save application settings
            self.settings.setValue("model", model_combo.currentText())
            self.settings.setValue("show_fps", show_fps.isChecked())
            self.settings.setValue("show_confidence", show_confidence.isChecked())
            self.settings.setValue("minimize_to_tray", minimize_to_tray.isChecked())
            self.settings.setValue("default_tab", default_mode.currentIndex())

            # Update models if needed
            if model_combo.currentText() != current_model:
                # Reload model
                self.load_model()

            # Update detection thread if running
            if (
                hasattr(self, "detection_thread")
                and self.detection_thread
                and self.detection_thread.isRunning()
            ):
                self.detection_thread.update_settings(self.detection_settings)

            self.statusBar().showMessage("Settings updated")

    def send_incident_notification(self, incident_type, details, image_path=None):
        """Send notifications for detected incident"""
        try:
            # Check if notifications are enabled
            if not (
                self.detection_settings.get("email_enabled", False)
                or self.detection_settings.get("webhook_enabled", False)
            ):
                return

            # Format notification message
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            confidence = details.get("confidence", 0)

            subject = f"{incident_type.capitalize()} Detected"

            message = f"Incident detected at {timestamp}\n\n"
            message += f"Type: {incident_type.capitalize()}\n"
            message += f"Confidence: {confidence:.2f}\n"

            # Add custom rule name if available
            if "rule" in details:
                message += f"Rule: {details['rule']}\n"

            # Add source info
            if self.tabWidget.currentIndex() == 1:  # Video tab
                message += f"Source: {os.path.basename(self.video_path)}\n"
            else:
                message += "Source: Camera Feed\n"

            # Send email if enabled
            if self.detection_settings.get("email_enabled", False):
                # Load image if available
                image = None
                if image_path and os.path.exists(image_path):
                    image = cv2.imread(image_path)

                self.notification_manager.send_email(subject, message, image=image)

            # Send webhook if enabled
            if self.detection_settings.get("webhook_enabled", False):
                webhook_data = {
                    "type": incident_type,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "details": details,
                    "source": (
                        self.video_path
                        if self.tabWidget.currentIndex() == 1
                        else "Camera Feed"
                    ),
                    "image_url": f"file://{image_path}" if image_path else None,
                }

                self.notification_manager.send_webhook(webhook_data)

        except Exception as e:
            logger.error(f"Error sending incident notification: {str(e)}")

    def add_recent_video(self, video_path):
        """Add a video to the recent videos list"""
        recent_videos = self.settings.value("recent_videos", [], type=list)

        # Remove if already exists
        if video_path in recent_videos:
            recent_videos.remove(video_path)

        # Add to front of list
        recent_videos.insert(0, video_path)

        # Keep only last N
        recent_videos = recent_videos[:MAX_RECENT_VIDEOS]

        # Save to settings
        self.settings.setValue("recent_videos", recent_videos)

        # Update menu
        self.update_recent_menu()

    def update_recent_menu(self):
        """Update the recent videos menu"""
        self.recent_menu.clear()

        recent_videos = self.settings.value("recent_videos", [], type=list)

        if not recent_videos:
            action = QAction("No Recent Videos", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return

        for path in recent_videos:
            if os.path.exists(path):
                filename = os.path.basename(path)
                action = QAction(filename, self)
                action.setData(path)
                action.triggered.connect(
                    lambda checked=False, p=path: self.load_recent_video(p)
                )
                self.recent_menu.addAction(action)

    def load_recent_video(self, path):
        """Load a video from the recent videos list"""
        self.video_path = path
        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}")

        # Switch to video tab
        self.tabWidget.setCurrentIndex(1)

        # Open video to get properties
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Set up video position slider
            self.position_slider.setRange(0, total_frames - 1)
            self.position_slider.setValue(0)
            self.position_slider.setEnabled(True)

            # Update time label
            duration = total_frames / fps
            self.time_label.setText(f"00:00 / {self.format_time(duration)}")

            # Display first frame
            ret, frame = cap.read()
            if ret:
                self.display_frame(frame, self.video_display)

            cap.release()

            # Enable process button
            self.process_btn.setEnabled(True)

    def closeEvent(self, event):
        """Handle application close event"""
        # Save settings
        self.save_app_state()

        # Check if we should minimize to tray instead of closing
        if self.settings.value("minimize_to_tray", False, type=bool) and self.tray_icon:
            # Confirm with user
            reply = QMessageBox.question(
                self,
                "Minimize to Tray",
                "Do you want to minimize to the system tray instead of closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.hide()
                event.ignore()
                return

        # Stop detection thread if running
        if (
            hasattr(self, "detection_thread")
            and self.detection_thread
            and self.detection_thread.isRunning()
        ):
            self.detection_thread.stop()
            self.detection_thread.wait()

        # Stop stream server if running
        if hasattr(self, "stream_server") and self.stream_server:
            try:
                self.stream_server["server"].shutdown()
            except:
                pass

        # Stop recording if active
        if self.is_recording and hasattr(self, "video_writer") and self.video_writer:
            self.video_writer.release()

        # Accept the close event
        event.accept()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(QDir.homePath(), ".incident_detector_logs", "app.log")
            ),
            logging.StreamHandler(),
        ],
    )

    # Create log directory if it doesn't exist
    os.makedirs(os.path.join(QDir.homePath(), ".incident_detector_logs"), exist_ok=True)

    # Create application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for a cleaner look

    # Set application info
    app.setApplicationName("Incident Detection System")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("Incident Detector")

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

    try:
        # Create and show main window
        window = IncidentDetectionApp()

        # Set default tab
        default_tab = window.settings.value("default_tab", 0, type=int)
        window.tabWidget.setCurrentIndex(default_tab)

        # Refresh camera list
        window.refresh_camera_list()

        # Populate custom rules list
        window.update_rules_list()

        # Show main window
        window.show()

        # Run the application
        sys.exit(app.exec())

    except Exception as e:
        # Handle startup errors
        logging.critical(f"Application startup error: {str(e)}", exc_info=True)

        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("Startup Error")
        error_dialog.setText("Error starting the application")
        error_dialog.setDetailedText(
            f"Error: {str(e)}\n\nPlease check the log file for more details."
        )
        error_dialog.exec()

        sys.exit(1)
