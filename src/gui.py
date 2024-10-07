import sys
import os
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QListWidget,
    QLabel,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QGroupBox,
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import src.main as main_module
from src import config


class FlagPredictionGUI(QMainWindow):
    """
    Main GUI class for the Flag Country Prediction and Similarity Analysis application.

    This class creates the main window and handles all user interactions, including
    loading flags, selecting query flags, training models, and displaying predictions.
    """

    def __init__(self):
        """
        Initialize the FlagPredictionGUI window and set up the user interface.
        """
        super().__init__()
        self.setWindowTitle("Flag Country Prediction and Similarity Analysis")
        self.setGeometry(100, 100, 1000, 800)

        # Initialize attributes for storing data and models
        self.flag_features = None  # Will store features of all flags
        self.query_flag = None  # Will store the currently selected flag
        self.model = None  # Will store the trained machine learning model
        self.label_encoder = None  # Will store the label encoder for country names

        self.init_ui()

    def init_ui(self):
        """
        Set up the user interface components and layouts.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Left side components
        self.load_button = QPushButton("Load Flags")
        self.load_button.clicked.connect(self.load_flags)
        left_layout.addWidget(self.load_button)

        self.flag_list = QListWidget()
        self.flag_list.itemClicked.connect(self.select_query_flag)
        left_layout.addWidget(self.flag_list)

        # Model selection components
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(config.MODELS_TO_TRAIN)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # Training options components
        training_group = QGroupBox("Training Options")
        training_layout = QVBoxLayout()
        self.multi_label_check = QCheckBox("Multi-label Classification")
        self.cross_validate_check = QCheckBox("Cross-validation")
        self.augmentation_spin = QSpinBox()
        self.augmentation_spin.setRange(1, 100)
        self.augmentation_spin.setValue(config.AUGMENTATION_FACTOR)
        self.augmentation_spin.setPrefix("Augmentation Factor: ")
        training_layout.addWidget(self.multi_label_check)
        training_layout.addWidget(self.cross_validate_check)
        training_layout.addWidget(self.augmentation_spin)
        training_group.setLayout(training_layout)
        left_layout.addWidget(training_group)

        # Training button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        left_layout.addWidget(self.train_button)

        # Right side components
        self.query_image = QLabel()
        self.query_image.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.query_image)

        self.prediction_label = QLabel()
        right_layout.addWidget(self.prediction_label)

        self.similar_flags_list = QListWidget()
        right_layout.addWidget(self.similar_flags_list)

        # Add layouts to main layout
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

    def load_flags(self):
        """
        Open a file dialog to select the directory containing flag images,
        process the flags, and populate the flag list.
        """
        flag_dir = QFileDialog.getExistingDirectory(self, "Select Flag Directory")
        if flag_dir:
            # Process flags and extract features
            self.flag_features = main_module.process_flags(flag_dir)
            # Clear and populate the flag list
            self.flag_list.clear()
            self.flag_list.addItems(self.flag_features.keys())
            # Load the pre-trained classifier
            self.model, self.label_encoder = main_module.load_classifier()

    def select_query_flag(self, item):
        """
        Handle the selection of a flag from the list.

        Args:
            item (QListWidgetItem): The selected item from the flag list.
        """
        self.query_flag = item.text()
        self.update_query_image()
        self.predict_country()
        self.find_similar_flags()

    def update_query_image(self):
        """
        Update the displayed image of the selected query flag.
        """
        if self.query_flag:
            pixmap = QPixmap(os.path.join("flag_images", self.query_flag))
            self.query_image.setPixmap(pixmap.scaled(300, 180, Qt.KeepAspectRatio))

    def predict_country(self):
        """
        Predict the country for the selected query flag and update the prediction label.
        """
        if self.query_flag and self.flag_features and self.model and self.label_encoder:
            query_features = self.flag_features[self.query_flag]
            predicted_country, confidence = main_module.predict_country(
                query_features,
                self.model,
                self.label_encoder,
                multi_label=self.multi_label_check.isChecked(),
            )
            if isinstance(predicted_country, list):
                countries = ", ".join(predicted_country)
                self.prediction_label.setText(f"Predicted countries: {countries}")
            else:
                self.prediction_label.setText(
                    f"Predicted country: {predicted_country} (confidence: {confidence:.2f})"
                )

    def find_similar_flags(self):
        """
        Find and display flags similar to the selected query flag.
        """
        if self.query_flag and self.flag_features:
            query_features = self.flag_features[self.query_flag]
            similar_flags = main_module.rank_flags(query_features, self.flag_features)

            self.similar_flags_list.clear()
            for name, similarity in similar_flags[1:11]:
                self.similar_flags_list.addItem(f"{name}: {similarity:.2f}")

    def train_model(self):
        """
        Train a new model based on the selected options and update the GUI.
        """
        # Get selected training options
        model_type = self.model_combo.currentText()
        multi_label = self.multi_label_check.isChecked()
        cross_validate = self.cross_validate_check.isChecked()
        augmentation_factor = self.augmentation_spin.value()

        # Update config with selected options
        config.MODELS_TO_TRAIN = [model_type]
        config.USE_MULTI_LABEL = multi_label
        config.USE_CROSS_VALIDATION = cross_validate
        config.AUGMENTATION_FACTOR = augmentation_factor

        # Call the training function from main_module
        main_module.train_models(
            multi_label=multi_label,
            cross_validate=cross_validate,
            models=[model_type],
            augmentation_factor=augmentation_factor,
        )

        # Reload the trained model
        self.model, self.label_encoder = main_module.load_classifier(
            model_type=model_type
        )

        self.prediction_label.setText("Model trained and loaded successfully!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlagPredictionGUI()
    window.show()
    sys.exit(app.exec())
