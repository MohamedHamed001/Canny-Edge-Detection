import sys
import threading
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QLineEdit, QGridLayout, QMessageBox, QStatusBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import cv2
import numpy as np
from functions import apply_canny

class EdgeDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.filePath = None
        self.initUI()
        
    def initUI(self):
        # Set window title and fixed size
        self.setWindowTitle('Improved Edge Detection - Task 4')
        self.setFixedSize(800, 500)
        
        # Adjust window flags to disable the maximize button
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        
        # Create main layout
        mainLayout = QVBoxLayout(self)
        
        # Create top buttons layout
        buttonsLayout = QHBoxLayout()
        self.browseButton = QPushButton('Browse', self)
        self.browseButton.clicked.connect(self.loadImage)
        buttonsLayout.addWidget(self.browseButton)
        
        self.applyButton = QPushButton('Apply', self)
        self.applyButton.clicked.connect(self.applyCanny)
        buttonsLayout.addWidget(self.applyButton)
        mainLayout.addLayout(buttonsLayout)
        
        # Create layout for images and their labels
        imagesLayout = QHBoxLayout()
        originalImageLayout = QVBoxLayout()
        
        # Original image title
        originalTitle = QLabel('Original')
        originalTitle.setAlignment(Qt.AlignCenter)
        originalTitle.setStyleSheet('font-weight: bold; font-size: 16px;')
        originalImageLayout.addWidget(originalTitle)
        
        # Original image label
        self.originalImageLabel = QLabel()
        self.originalImageLabel.setAlignment(Qt.AlignCenter)
        self.originalImageLabel.setFixedSize(400, 250)
        originalImageLayout.addWidget(self.originalImageLabel)
        
        imagesLayout.addLayout(originalImageLayout)
        
        resultImageLayout = QVBoxLayout()
        
        # Result image title
        resultTitle = QLabel('Result')
        resultTitle.setAlignment(Qt.AlignCenter)
        resultTitle.setStyleSheet('font-weight: bold; font-size: 16px;')
        resultImageLayout.addWidget(resultTitle)
        
        # Result image label
        self.resultImageLabel = QLabel()
        self.resultImageLabel.setAlignment(Qt.AlignCenter)
        self.resultImageLabel.setFixedSize(400, 250)
        resultImageLayout.addWidget(self.resultImageLabel)
        
        imagesLayout.addLayout(resultImageLayout)
        mainLayout.addLayout(imagesLayout)
        
        # Create layout for inputs with titles
        inputsLayout = QGridLayout()
        
        # Sigma input and title
        sigmaTitle = QLabel('Sigma')
        sigmaTitle.setAlignment(Qt.AlignCenter)
        inputsLayout.addWidget(sigmaTitle, 0, 0)
        self.sigmaInput = QLineEdit('1.0')
        inputsLayout.addWidget(self.sigmaInput, 1, 0)
        
        # Kernel size input and title
        kernelSizeTitle = QLabel('Kernel size')
        kernelSizeTitle.setAlignment(Qt.AlignCenter)
        inputsLayout.addWidget(kernelSizeTitle, 0, 1)
        self.kernelSizeInput = QLineEdit('3')
        inputsLayout.addWidget(self.kernelSizeInput, 1, 1)
        
        # Low threshold input and title
        lowThresholdTitle = QLabel('Low Threshold')
        lowThresholdTitle.setAlignment(Qt.AlignCenter)
        inputsLayout.addWidget(lowThresholdTitle, 2, 0)
        self.lowThresholdInput = QLineEdit('50')
        inputsLayout.addWidget(self.lowThresholdInput, 3, 0)
        
        # High threshold input and title
        highThresholdTitle = QLabel('High Threshold')
        highThresholdTitle.setAlignment(Qt.AlignCenter)
        inputsLayout.addWidget(highThresholdTitle, 2, 1)
        self.highThresholdInput = QLineEdit('150')
        inputsLayout.addWidget(self.highThresholdInput, 3, 1)
        
        mainLayout.addLayout(inputsLayout)
        
        # Status bar setup
        self.statusBar = QStatusBar()
        mainLayout.addWidget(self.statusBar)
    def loadImage(self):
        self.filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if self.filePath:
            image = cv2.imread(self.filePath, cv2.IMREAD_COLOR)
            if image is not None:
                self.displayImage(image, self.originalImageLabel)


        
    def applyCanny(self):
        if not self.filePath:
            QMessageBox.warning(self, 'Warning', 'Please select an image first!')
            return
        # This should be implemented based on your functions.py and needs the actual parameters to be fetched
        self.statusBar.showMessage("Processing image...")
        QApplication.processEvents()
        sigma = float(self.sigmaInput.text() or 1.0)
        kernel_size = int(self.kernelSizeInput.text() or 3)
        low_threshold = int(self.lowThresholdInput.text() or 50)
        high_threshold = int(self.highThresholdInput.text() or 150)
        threading.Thread(target=self.startCannyProcessing, args=(sigma, kernel_size, low_threshold, high_threshold), daemon=True).start()

    def startCannyProcessing(self, sigma, kernel_size, low_threshold, high_threshold):
        final_edges = apply_canny(self.filePath, sigma, kernel_size, low_threshold, high_threshold)
        
        # Update UI from the main thread
        self.displayImage(final_edges, self.resultImageLabel, convert_color=False)

        # Signal that processing is complete
        self.statusBar.showMessage("Image processing complete.", 5000)




    def displayImage(self, cvImg, label, convert_color=True):
        # Check if the image is color or grayscale for correct handling
        if len(cvImg.shape) == 3:  # Color image
            height, width, channel = cvImg.shape
            bytesPerLine = 3 * width
            if convert_color:
                cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
            qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        else:  # Grayscale image
            height, width = cvImg.shape
            bytesPerLine = width
            qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))


def main():
    app = QApplication(sys.argv)
    ex = EdgeDetectionApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
