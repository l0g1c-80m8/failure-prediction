import sys
import os
import glob
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QSlider, QSpinBox, QDoubleSpinBox,
                             QSplitter, QGridLayout, QGroupBox, QStatusBar, QComboBox, QDialog, 
                             QFormLayout, QDialogButtonBox, QListWidget, QMenu, QAction,
                             QSizePolicy, QTextEdit)
from PyQt5.QtCore import Qt, QSize
from widgets import (ContourImageWidget, ValuePlotter, FileListDialog)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the function from common_functions
from common_functions import (piecewise_bezier_interpolation, spline_interpolation, hermite_spline_interpolation)

# Set environment variable to avoid Qt conflicts
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""


class EpisodeEditor(QMainWindow):
    """Main window for episode editing"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Episode Data Editor")
        self.setGeometry(100, 100, 1200, 800)
        
        self.episode_data = None
        self.current_step = 0
        self.current_file_path = None
        self.current_dir_path = None
        self.file_list = []
        self.current_file_index = -1
        self.original_risk_values = None
        
        self.init_ui()
        
        # Create context menu for right-click
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def show_context_menu(self, pos):
        """Show context menu with additional options"""
        context_menu = QMenu(self)
        
        # Add actions
        load_file_action = QAction("Load File", self)
        load_file_action.triggered.connect(self.load_episode_data)
        
        load_folder_action = QAction("Load Folder", self)
        load_folder_action.triggered.connect(self.load_folder)
        
        # Add actions to menu
        context_menu.addAction(load_file_action)
        context_menu.addAction(load_folder_action)
        
        # Show the menu
        context_menu.exec_(self.mapToGlobal(pos))
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # File info and controls
        file_info_layout = QHBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("font-weight: bold;")
        file_info_layout.addWidget(self.file_label)
        file_info_layout.addStretch()
        
        # File navigation controls (for folder mode)
        self.prev_file_button = QPushButton("← Previous File")
        self.prev_file_button.clicked.connect(self.previous_file)
        self.prev_file_button.setEnabled(False)
        
        self.next_file_button = QPushButton("Next File →")
        self.next_file_button.clicked.connect(self.next_file)
        self.next_file_button.setEnabled(False)
        
        file_info_layout.addWidget(self.prev_file_button)
        file_info_layout.addWidget(self.next_file_button)
        
        main_layout.addLayout(file_info_layout)
        
        # File controls
        file_controls = QHBoxLayout()
        self.load_file_button = QPushButton("Load File")
        self.load_file_button.clicked.connect(self.load_episode_data)
        
        self.load_folder_button = QPushButton("Load Folder")
        self.load_folder_button.clicked.connect(self.load_folder)
        
        self.save_button = QPushButton("Save Modified Data")
        self.save_button.clicked.connect(self.save_episode_data)
        self.save_button.setEnabled(False)
        
        # Add restore button
        self.restore_button = QPushButton("Restore Original Values")
        self.restore_button.clicked.connect(self.restore_original_risk_values)
        self.restore_button.setEnabled(False)

        file_controls.addWidget(self.load_file_button)
        file_controls.addWidget(self.load_folder_button)
        file_controls.addWidget(self.save_button)
        file_controls.addWidget(self.restore_button)
        main_layout.addLayout(file_controls)
        
        # Split view: top for images, bottom for controls
        splitter = QSplitter(Qt.Vertical)
        
        # Image display area
        image_widget = QWidget()
        image_layout = QGridLayout()
        image_layout.setSpacing(10)  # Add some spacing between elements
        
        # Top camera views
        top_group = QGroupBox("Top Camera View")
        top_layout = QHBoxLayout()
        self.top_panel_view = ContourImageWidget()
        self.top_object_view = ContourImageWidget()
        top_layout.addWidget(self.top_panel_view)
        top_layout.addWidget(self.top_object_view)
        top_group.setLayout(top_layout)
        
        # Front camera views
        front_group = QGroupBox("Front Camera View")
        front_layout = QHBoxLayout()
        self.front_panel_view = ContourImageWidget()
        self.front_object_view = ContourImageWidget()
        front_layout.addWidget(self.front_panel_view)
        front_layout.addWidget(self.front_object_view)
        front_group.setLayout(front_layout)
        
        image_layout.addWidget(top_group, 0, 0)
        image_layout.addWidget(front_group, 0, 1)
        
        # Value plot
        self.value_plot = ValuePlotter()
        image_layout.addWidget(self.value_plot, 1, 0, 1, 2)
        
        # New: State values display
        state_group = QGroupBox("State Values")
        state_layout = QVBoxLayout()
        self.state_display = QTextEdit()
        self.state_display.setReadOnly(True)
        self.state_display.setFixedHeight(150)
        state_layout.addWidget(self.state_display)
        state_group.setLayout(state_layout)
        image_layout.addWidget(state_group, 2, 0, 1, 2)
        
        image_widget.setLayout(image_layout)
        splitter.addWidget(image_widget)
        
        # Controls area
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Step")
        self.prev_button.clicked.connect(self.previous_step)
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setTickPosition(QSlider.TicksBelow)
        self.step_slider.valueChanged.connect(self.slider_changed)
        self.next_button = QPushButton("Next Step")
        self.next_button.clicked.connect(self.next_step)
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.step_slider)
        nav_layout.addWidget(self.next_button)
        
        # Step info
        step_info_layout = QHBoxLayout()
        step_info_layout.addWidget(QLabel("Current Step:"))
        self.step_spin = QSpinBox()
        self.step_spin.valueChanged.connect(self.spinbox_changed)
        step_info_layout.addWidget(self.step_spin)
        
        step_info_layout.addWidget(QLabel("Total Steps:"))
        self.total_steps_label = QLabel("0")
        step_info_layout.addWidget(self.total_steps_label)
        
        step_info_layout.addStretch()
        
        # Value editing
        value_edit_layout = QHBoxLayout()
        self.value_label = QLabel("Risk:")
        value_edit_layout.addWidget(self.value_label)
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(0.0, 1.0)
        self.value_spin.setSingleStep(0.1)
        self.value_spin.valueChanged.connect(self.value_changed)
        value_edit_layout.addWidget(self.value_spin)
        
        # Add new button for applying the markers
        self.apply_markers_button = QPushButton("Apply Risk Markers")
        self.apply_markers_button.clicked.connect(self.apply_risk_markers)
        self.apply_markers_button.setEnabled(False)
        value_edit_layout.addWidget(self.apply_markers_button)
        
        self.interpolate_button = QPushButton("Interpolate Values")
        self.interpolate_button.clicked.connect(self.interpolate_values)
        value_edit_layout.addWidget(self.interpolate_button)
        
        value_edit_layout.addStretch()
        
        # Add instructions label
        instructions_label = QLabel("Click on plot to set first failure marker (green), click again to set failure complete marker (red). Then press 'Apply Risk Markers'.")
        instructions_label.setStyleSheet("color: #555; font-style: italic;")
        
        # Add all layouts to controls
        controls_layout.addLayout(nav_layout)
        controls_layout.addLayout(step_info_layout)
        controls_layout.addLayout(value_edit_layout)
        controls_layout.addWidget(instructions_label)
        
        controls_widget.setLayout(controls_layout)
        splitter.addWidget(controls_widget)
        
        # Set initial sizes for splitter
        splitter.setSizes([600, 200])
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Disable controls initially
        self.set_controls_enabled(False)
    
    def set_controls_enabled(self, enabled):
        """Enable or disable controls based on data availability"""
        self.prev_button.setEnabled(enabled)
        self.step_slider.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.step_spin.setEnabled(enabled)
        self.value_spin.setEnabled(enabled)
        self.interpolate_button.setEnabled(enabled)
        self.apply_markers_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.restore_button.setEnabled(enabled and self.original_risk_values is not None)
    
    def load_folder(self):
        """Load a folder of .npy files"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with Episode Data", "")
        
        if not folder_path:
            return
            
        # Find all .npy files in the folder
        file_pattern = os.path.join(folder_path, "*.npy")
        file_paths = glob.glob(file_pattern)
        
        if not file_paths:
            self.status_bar.showMessage("No .npy files found in the selected folder")
            return
        
        # Store file list and current directory
        self.file_list = sorted(file_paths)
        self.current_dir_path = folder_path
        
        # Show dialog to select a file
        dialog = FileListDialog(self.file_list, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_file is not None:
            selected_file = dialog.selected_file
            selected_path = os.path.join(folder_path, selected_file)
            
            # Find index of selected file
            self.current_file_index = self.file_list.index(
                next((f for f in self.file_list if os.path.basename(f) == selected_file), None))
            
            # Enable file navigation buttons
            self.prev_file_button.setEnabled(self.current_file_index > 0)
            self.next_file_button.setEnabled(self.current_file_index < len(self.file_list) - 1)
            
            # Load the selected file
            self.load_file(selected_path)
        else:
            self.status_bar.showMessage("No file selected")
    
    def previous_file(self):
        """Load the previous file in the folder"""
        if self.current_file_index > 0:
            self.current_file_index -= 1
            file_path = self.file_list[self.current_file_index]
            self.load_file(file_path)
            
            # Update button state
            self.prev_file_button.setEnabled(self.current_file_index > 0)
            self.next_file_button.setEnabled(self.current_file_index < len(self.file_list) - 1)
    
    def next_file(self):
        """Load the next file in the folder"""
        if self.current_file_index < len(self.file_list) - 1:
            self.current_file_index += 1
            file_path = self.file_list[self.current_file_index]
            self.load_file(file_path)
            
            # Update button state
            self.prev_file_button.setEnabled(self.current_file_index > 0)
            self.next_file_button.setEnabled(self.current_file_index < len(self.file_list) - 1)
    
    def load_episode_data(self):
        """Load episode data from a .npy file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Episode Data", "", "NumPy Files (*.npy)")
        
        if not file_path:
            return
        
        # Clear file navigation state
        self.file_list = []
        self.current_file_index = -1
        self.prev_file_button.setEnabled(False)
        self.next_file_button.setEnabled(False)
        
        self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load data from the specified file path"""
        try:
            self.status_bar.showMessage(f"Loading data from {os.path.basename(file_path)}...")
            self.episode_data = np.load(file_path, allow_pickle=True)
            
            # Set current file path
            self.current_file_path = file_path
            self.file_label.setText(f"File: {os.path.basename(file_path)}")
            
            # Check if 'risk' key exists in data
            if 'risk' not in self.episode_data[0]:
                self.status_bar.showMessage("Warning: 'risk' key not found in data")
                return
            
            # Save the original risk values
            self.original_risk_values = [np.copy(step['risk']) for step in self.episode_data]
                                     
            # Set up controls
            num_steps = len(self.episode_data)
            self.total_steps_label.setText(str(num_steps))
            
            self.step_slider.setRange(0, num_steps - 1)
            self.step_slider.setSingleStep(1)
            self.step_slider.setPageStep(10)
            
            self.step_spin.setRange(0, num_steps - 1)
            
            # Extract risk values for plotting
            risk_values = [step['risk'][0] for step in self.episode_data]
            self.value_plot.set_data(risk_values)
            
            # Reset markers
            self.value_plot.set_markers(None, None)
            
            # Enable controls
            self.set_controls_enabled(True)
            
            # Show first step
            self.current_step = 0
            self.update_display()
            
            self.status_bar.showMessage(f"Loaded {num_steps} steps from {os.path.basename(file_path)}")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading file: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def restore_original_risk_values(self):
        """Restore the risk values to their original state when the file was loaded"""
        if self.episode_data is None or self.original_risk_values is None:
            self.status_bar.showMessage("No data loaded or no original values to restore")
            return
        
        try:
            # Restore the original risk values
            for i, original_risk in enumerate(self.original_risk_values):
                self.episode_data[i]['risk'] = np.copy(original_risk)
            
            # Update the plot
            risk_values = [step['risk'][0] for step in self.episode_data]
            self.value_plot.set_data(risk_values)
            
            # Update display
            self.update_display()
            
            self.status_bar.showMessage("Restored original risk values")
        except Exception as e:
            self.status_bar.showMessage(f"Error restoring original values: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_episode_data(self):
        """Save the modified episode data"""
        if self.episode_data is None:
            return
            
        # Suggest the current file name as default
        default_path = ""
        if self.current_file_path:
            dirname = os.path.dirname(self.current_file_path)
            basename = os.path.basename(self.current_file_path)
            name, ext = os.path.splitext(basename)
            default_path = os.path.join(dirname, f"{name}_modified{ext}")
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Modified Data", default_path, "NumPy Files (*.npy)")
        
        if not file_path:
            return
            
        try:
            np.save(file_path, self.episode_data)
            self.status_bar.showMessage(f"Saved data to {os.path.basename(file_path)}")
        except Exception as e:
            self.status_bar.showMessage(f"Error saving file: {str(e)}")
    
    def update_display(self):
        """Update display with current step data"""
        if self.episode_data is None or not 0 <= self.current_step < len(self.episode_data):
            return
            
        step_data = self.episode_data[self.current_step]
        
        # Update images and contours
        if 'full_top_frame_rgb' in step_data and step_data['full_top_frame_rgb'] is not None:
            self.top_panel_view.set_image(step_data['full_top_frame_rgb'])
            self.top_panel_view.set_contour(step_data['gripper_top_contour'])
        else:
            self.top_panel_view.clear()
        
        if 'full_top_frame_rgb' in step_data and step_data['full_top_frame_rgb'] is not None:
            self.top_object_view.set_image(step_data['full_top_frame_rgb'])
            self.top_object_view.set_contour(step_data['object_top_contour'])
        else:
            self.top_object_view.clear()
        
        if 'full_front_frame_rgb' in step_data and step_data['full_front_frame_rgb'] is not None:
            self.front_panel_view.set_image(step_data['full_front_frame_rgb'])
            self.front_panel_view.set_contour(step_data['gripper_front_contour'])
        else:
            self.front_panel_view.clear()
        
        if 'full_front_frame_rgb' in step_data and step_data['full_front_frame_rgb'] is not None:
            self.front_object_view.set_image(step_data['full_front_frame_rgb'])
            self.front_object_view.set_contour(step_data['object_front_contour'])
        else:
            self.front_object_view.clear()
        
        # Update controls to reflect current values
        self.step_slider.blockSignals(True)
        self.step_slider.setValue(self.current_step)
        self.step_slider.blockSignals(False)
        
        self.step_spin.blockSignals(True)
        self.step_spin.setValue(self.current_step)
        self.step_spin.blockSignals(False)
        
        # Update risk value
        if 'risk' in step_data:
            risk_value = step_data['risk'][0]
            self.value_spin.blockSignals(True)
            self.value_spin.setValue(risk_value)
            self.value_spin.blockSignals(False)
            
            # Update plot to highlight current point
            self.value_plot.set_current_index(self.current_step)
            
            # Update status
            self.status_bar.showMessage(f"Step {self.current_step}, Risk: {risk_value:.2f}")
        else:
            self.status_bar.showMessage(f"Step {self.current_step}, 'risk' key not found")
        
        # Update state values display
        if 'state' in step_data:
            state_values = step_data['state']
            state_text = f"State Values (Step {self.current_step}):\n"
            
            if isinstance(state_values, np.ndarray):
                # Format each value with index and format for readability
                for i, val in enumerate(state_values):
                    state_text += f"[{i}]: {val:.4f}\n"
            else:
                state_text += f"{state_values}"
                
            self.state_display.setText(state_text)
        else:
            self.state_display.setText(f"No state data available for step {self.current_step}")
    
    def previous_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()
    
    def next_step(self):
        """Go to next step"""
        if self.episode_data is not None and self.current_step < len(self.episode_data) - 1:
            self.current_step += 1
            self.update_display()
    
    def slider_changed(self, value):
        """Handle slider value change"""
        if value != self.current_step:
            self.current_step = value
            self.update_display()
    
    def spinbox_changed(self, value):
        """Handle step spinbox value change"""
        if value != self.current_step:
            self.current_step = value
            self.update_display()
    
    def value_changed(self, value):
        """Update the risk value for the current step"""
        if self.episode_data is None:
            return
            
        # Update the value in the data
        self.episode_data[self.current_step]['risk'] = np.asarray([value], dtype=np.float32)
        
        # Update the plot
        risk_values = [step['risk'][0] for step in self.episode_data]
        self.value_plot.set_data(risk_values)
        
        self.status_bar.showMessage(f"Updated step {self.current_step} risk to {value:.2f}")
    
    def interpolate_values(self):
        """Open dialog to interpolate values between two points"""
        if self.episode_data is None:
            return
            
        # Create dialog window
        dialog = QDialog(self)
        dialog.setWindowTitle("Interpolate Risk Values")
        
        dialog_layout = QFormLayout()
        
        # Start step
        start_spin = QSpinBox()
        start_spin.setRange(0, len(self.episode_data) - 1)
        start_spin.setValue(max(0, self.current_step - 10))
        dialog_layout.addRow("Start Step:", start_spin)
        
        # Start value
        start_value_spin = QDoubleSpinBox()
        start_value_spin.setRange(0.0, 1.0)
        start_value_spin.setSingleStep(0.1)
        start_value_spin.setValue(0.0)
        dialog_layout.addRow("Start Value:", start_value_spin)
        
        # End step
        end_spin = QSpinBox()
        end_spin.setRange(0, len(self.episode_data) - 1)
        end_spin.setValue(min(len(self.episode_data) - 1, self.current_step + 10))
        dialog_layout.addRow("End Step:", end_spin)
        
        # End value
        end_value_spin = QDoubleSpinBox()
        end_value_spin.setRange(0.0, 1.0)
        end_value_spin.setSingleStep(0.1)
        end_value_spin.setValue(1.0)
        dialog_layout.addRow("End Value:", end_value_spin)
        
        # Interpolation type
        interp_combo = QComboBox()
        interp_combo.addItems(["Linear", "Step", "Smooth"])
        dialog_layout.addRow("Interpolation Type:", interp_combo)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addRow(button_box)
        
        dialog.setLayout(dialog_layout)
        
        # Handle dialog result
        if dialog.exec_() == QDialog.Accepted:
            start_step = start_spin.value()
            start_value = start_value_spin.value()
            end_step = end_spin.value()
            end_value = end_value_spin.value()
            interp_type = interp_combo.currentText()
            
            # Apply interpolation
            self.apply_interpolation(start_step, start_value, end_step, end_value, interp_type)
    
    def apply_interpolation(self, start_step, start_value, end_step, end_value, interp_type):
        """Apply interpolation between two points"""
        if start_step >= end_step:
            self.status_bar.showMessage("Error: Start step must be less than end step")
            return
            
        # Generate interpolated values
        num_steps = end_step - start_step + 1
        
        if interp_type == "Linear":
            # Linear interpolation
            values = np.linspace(start_value, end_value, num_steps)
        elif interp_type == "Step":
            # Step function
            values = np.zeros(num_steps)
            mid_point = num_steps // 2
            values[:mid_point] = start_value
            values[mid_point:] = end_value
        else:  # Smooth
            # Smooth S-curve interpolation
            t = np.linspace(0, 1, num_steps)
            # Use cubic function for smoother transition
            values = start_value + (end_value - start_value) * (3 * t**2 - 2 * t**3)
        
        # Apply values to the data
        for i, value in enumerate(values, start=start_step):
            # print("value", value)
            self.episode_data[i]["risk"] = np.asarray([value], dtype=np.float32)
        
        # Update plot
        risk_values = [step['risk'][0] for step in self.episode_data]
        self.value_plot.set_data(risk_values)
        
        # Update current display
        self.update_display()
        
        self.status_bar.showMessage(f"Applied {interp_type} interpolation from step {start_step} to {end_step}")
    
    def resizeEvent(self, event):
        """Override resize event to ensure image widgets maintain fixed size"""
        super().resizeEvent(event)
        # Force image widgets to maintain their fixed size
        self.top_panel_view.setFixedSize(320, 240)
        self.top_object_view.setFixedSize(320, 240)
        self.front_panel_view.setFixedSize(320, 240)
        self.front_object_view.setFixedSize(320, 240)

    def apply_risk_markers(self):
        """Apply risk values based on the marker positions using advanced interpolation"""
        if self.episode_data is None:
            return
        
        # Get marker positions
        first_failure, failure_trim = self.value_plot.get_markers()
        
        if first_failure is None:
            self.status_bar.showMessage("Error: First failure marker not set")
            return
            
        if failure_trim is None:
            self.status_bar.showMessage("Error: Failure complete marker not set")
            return
            
        if first_failure >= failure_trim:
            self.status_bar.showMessage("Error: First failure must be before failure complete")
            return
        
        try:
            # Create interpolation dialog to let user choose the interpolation method
            dialog = QDialog(self)
            dialog.setWindowTitle("Choose Interpolation Method")
            
            dialog_layout = QFormLayout()
            
            # Interpolation type
            interp_combo = QComboBox()
            interp_combo.addItems(["Linear", "Bezier", "Cubic Spline", "Hermite Spline"])
            dialog_layout.addRow("Interpolation Type:", interp_combo)
            
            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            dialog_layout.addRow(button_box)
            
            dialog.setLayout(dialog_layout)
            
            # Handle dialog result
            if dialog.exec_() == QDialog.Accepted:
                interp_type = interp_combo.currentText()
                
                # Set all values before first_failure to 0
                for i in range(first_failure):
                    self.episode_data[i]['risk'] = np.asarray([0.0], dtype=np.float32)
                
                # Map our markers correctly:
                # 0 = start_idx (beginning of the episode with value 0.0)
                # first_failure = middle_idx (point with value 0.5)
                # failure_trim = end_idx (point with value 1.0)
                start_idx = 0
                middle_idx = first_failure
                end_idx = failure_trim
                
                # Get interpolated values based on selected method
                if interp_type == "Linear":
                    # Simple linear interpolation
                    interpolated = np.linspace(0, 1, failure_trim + 1)
                elif interp_type == "Bezier":
                    interpolated = piecewise_bezier_interpolation(start_idx, middle_idx, end_idx)
                elif interp_type == "Cubic Spline":
                    interpolated = spline_interpolation(start_idx, middle_idx, end_idx)
                elif interp_type == "Hermite Spline":
                    interpolated = hermite_spline_interpolation(start_idx, middle_idx, end_idx)
                
                # Apply interpolated values (entire range from 0 to failure_trim)
                for i, value in enumerate(interpolated):
                    if i < len(self.episode_data):  # Ensure we don't go out of bounds
                        self.episode_data[i]['risk'] = np.asarray([value], dtype=np.float32)
                
                # Set all values after failure_trim to 1
                for i in range(failure_trim + 1, len(self.episode_data)):
                    self.episode_data[i]['risk'] = np.asarray([1.0], dtype=np.float32)
                
                # Update plot with new values
                risk_values = [step['risk'][0] for step in self.episode_data]
                self.value_plot.set_data(risk_values)
                
                # Update display
                self.update_display()
                
                self.status_bar.showMessage(f"Applied {interp_type} interpolation with {first_failure} as midpoint and {failure_trim} as endpoint")
        except Exception as e:
            self.status_bar.showMessage(f"Error applying risk values: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    editor = EpisodeEditor()
    editor.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()