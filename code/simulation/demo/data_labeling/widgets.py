

from PyQt5.QtWidgets import (QVBoxLayout, QDialog,
                             QLabel, QDialogButtonBox, QListWidget, 
                             QSizePolicy)
from PyQt5.QtCore import Qt
import os
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
# Force matplotlib to use a different backend to avoid conflicts
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

class ContourImageWidget(QLabel):
    """Widget to display an image with contours overlaid"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set fixed size for the image displays
        self.setFixedSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #cccccc;")
        # Disable size adjustments
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.original_image = None
        self.contour = None
        
    def set_image(self, image):
        """Set the background image"""
        if image is None:
            self.clear()
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_image = image
        self.update_display()
        
    def set_contour(self, contour):
        """Set the contour to overlay"""
        self.contour = contour
        self.update_display()
        
    def update_display(self):
        """Update the display with current image and contour"""
        if self.original_image is None:
            return
            
        # Make a copy of the image to draw on
        display_img = self.original_image.copy()
        
        # Draw contours if they exist
        if self.contour is not None and len(self.contour) > 0:
            # Convert to list of points for OpenCV
            contour_points = self.contour.reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(display_img, [contour_points], 0, (0, 255, 0), 2)
        
        # Convert to QImage and display
        height, width, channel = display_img.shape
        bytes_per_line = 3 * width
        
        # We need to handle different OpenCV color order
        # Convert BGR to RGB if needed
        if len(display_img.shape) == 3 and display_img.shape[2] == 3:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
        q_img = QImage(display_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale the pixmap to fit our fixed size but maintain aspect ratio
        pixmap = pixmap.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        
    def resizeEvent(self, event):
        """Override resize event to maintain fixed size"""
        self.setFixedSize(320, 240)
        super().resizeEvent(event)


class ValuePlotter(FigureCanvas):
    """Widget for plotting risk values with draggable markers"""
    
    def __init__(self, ylabel, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        # Set minimum size for the plot but allow it to expand horizontally
        self.setMinimumSize(600, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.values = []
        self.current_index = -1
        
        # Add markers for risk transition
        self.first_failure_time_step = None
        self.failure_time_step_trim = None
        
        # Connect mouse events
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        
        # Variables for dragging
        self.dragging = False
        self.active_marker = None

        self.ylabel = ylabel
        
    def set_data(self, values):
        """Set the data to plot"""
        self.values = values
        self.update_plot()
        
    def set_current_index(self, index):
        """Highlight the current index"""
        self.current_index = index
        self.update_plot()
        
    def set_markers(self, first_failure, failure_trim):
        """Set the risk transition markers"""
        self.first_failure_time_step = first_failure
        self.failure_time_step_trim = failure_trim
        self.update_plot()
        
    def get_markers(self):
        """Get the current marker positions"""
        return self.first_failure_time_step, self.failure_time_step_trim
        
    def update_plot(self):
        """Update the plot with current data"""
        self.ax.clear()
        if len(self.values) > 0:
            time_steps = range(len(self.values))
            self.ax.plot(time_steps, self.values, 'b-')
            
            # Highlight current point if valid
            if 0 <= self.current_index < len(self.values):
                self.ax.plot(self.current_index, self.values[self.current_index], 'ro')
            
            # Draw failure markers if set
            if self.first_failure_time_step is not None:
                self.ax.axvline(x=self.first_failure_time_step, color='g', linestyle='--', 
                               alpha=0.7, label='First Failure')
            
            if self.failure_time_step_trim is not None:
                self.ax.axvline(x=self.failure_time_step_trim, color='r', linestyle='--', 
                               alpha=0.7, label='Failure Complete')
            
            self.ax.set_ylim(-0.1, 1.1)  # Assuming values are between 0 and 1
            self.ax.set_xlabel('Time Step')
            self.ax.set_ylabel(self.ylabel)
            self.ax.grid(True)
            
            if self.first_failure_time_step is not None or self.failure_time_step_trim is not None:
                self.ax.legend()
        
        self.fig.tight_layout()
        self.draw()
    
    def set_state_data(self, state_data):
        """Set the state data to plot as multiple series"""
        if state_data is None or len(state_data) == 0:
            return
            
        self.ax.clear()
        
        # Plot each dimension of state data with a different color
        num_dimensions = len(state_data[0])
        for i in range(num_dimensions):
            dimension_values = [step[i] for step in state_data]
            self.ax.plot(range(len(dimension_values)), dimension_values, 
                        label=f'Dim {i}')
        
        # Highlight current point if valid
        if 0 <= self.current_index < len(state_data):
            for i in range(num_dimensions):
                self.ax.plot(self.current_index, state_data[self.current_index][i], 'ro', markersize=4)
        
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True)
        
        # Add a legend, but only for a reasonable number of dimensions
        if num_dimensions <= 10:  # Don't show legend for too many dimensions
            self.ax.legend(loc='upper right', fontsize='small')
        
        self.fig.tight_layout()
        self.draw()

    def on_click(self, event):
        """Handle mouse click to set or drag markers"""
        if event.inaxes != self.ax or event.button != 1:
            return
            
        if event.xdata is None:
            return
            
        x_pos = int(round(event.xdata))
        if x_pos < 0 or x_pos >= len(self.values):
            return
        
        # Check if click is near existing markers
        if self.first_failure_time_step is not None and abs(x_pos - self.first_failure_time_step) < 5:
            self.dragging = True
            self.active_marker = 'first'
            return
            
        if self.failure_time_step_trim is not None and abs(x_pos - self.failure_time_step_trim) < 5:
            self.dragging = True
            self.active_marker = 'second'
            return
        
        # Set new marker
        if self.first_failure_time_step is None:
            self.first_failure_time_step = x_pos
        elif self.failure_time_step_trim is None:
            # Ensure second marker is after first
            if x_pos > self.first_failure_time_step:
                self.failure_time_step_trim = x_pos
        else:
            # Reset markers and start over
            self.first_failure_time_step = x_pos
            self.failure_time_step_trim = None
        
        self.update_plot()
        
    def on_motion(self, event):
        """Handle mouse drag to move markers"""
        if not self.dragging or event.xdata is None:
            return
            
        x_pos = int(round(event.xdata))
        if x_pos < 0 or x_pos >= len(self.values):
            return
            
        if self.active_marker == 'first':
            # First marker can't be after second
            if self.failure_time_step_trim is None or x_pos < self.failure_time_step_trim:
                self.first_failure_time_step = x_pos
        elif self.active_marker == 'second':
            # Second marker can't be before first
            if x_pos > self.first_failure_time_step:
                self.failure_time_step_trim = x_pos
                
        self.update_plot()
        
    def on_release(self, event):
        """Handle mouse release to stop dragging"""
        self.dragging = False
        self.active_marker = None


class FileListDialog(QDialog):
    """Dialog to select a file from a list"""
    
    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select a File")
        self.setGeometry(300, 300, 500, 400)
        
        self.selected_file = None
        
        # Layout
        layout = QVBoxLayout()
        
        # File list
        self.file_list = QListWidget()
        
        # Add files to the list
        for file_path in file_paths:
            self.file_list.addItem(os.path.basename(file_path))
        
        # Connect double click to accept
        self.file_list.itemDoubleClicked.connect(self.accept)
        
        layout.addWidget(QLabel("Double-click a file to open:"))
        layout.addWidget(self.file_list)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def on_accept(self):
        """Handle accept button"""
        if self.file_list.currentItem():
            self.selected_file = self.file_list.currentItem().text()
            self.accept()
        else:
            # No file selected
            pass
