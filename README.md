
# Improved Edge Detection

## Project Overview
This project implements an improved edge detection tool using PyQt for the interface and functions made from scratch for the image processing. The application allows users to upload images and apply Canny edge detection with customizable parameters for sigma, kernel size, and threshold values, providing flexibility in how edge detection is performed.

## Features
- **Interactive GUI**: Built with PyQt, the interface includes image display panels, parameter input fields, and buttons to load images and apply edge detection.
- **Custom Edge Detection**: Users can adjust sigma, kernel size, low threshold, and high threshold to fine-tune the edge detection process.
- **Real-time Image Processing**: Displays the original and processed images side-by-side for easy comparison.

## Technologies Used
- Python 3
- PyQt5
- OpenCV
- NumPy
- PIL

## Installation
Ensure Python 3.x is installed on your system. You can then install the necessary libraries using pip:

```bash
pip install pyqt5 opencv-python-headless numpy pillow
```

## How to Run
To start the application, navigate to the directory containing the script and run:

```bash
python interface.py
```

## Usage
1. **Start the application** as described above.
2. **Load an Image**: Click the 'Browse' button to open a file dialog where you can select an image.
3. **Set Parameters**: Input desired values for sigma, kernel size, and thresholds or use the default values.
4. **Apply Edge Detection**: Click the 'Apply' button to process the image. The results will be displayed in the 'Result' panel.

## Application Structure
- `interface.py`: Handles the GUI and user interactions.
- `functions.py`: Contains all the image processing functions, including Gaussian blurring, Sobel filters, non-maximum suppression, double thresholding, and hysteresis for edge detection.

## Example Images
![Tropical Wave Pattern](images\results\result.png)


## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Authors
- Mohamed Hamed



