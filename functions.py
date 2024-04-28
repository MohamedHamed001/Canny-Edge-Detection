import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
def gaussian_kernel(size, sigma):
    """ Generate a Gaussian kernel matrix. """
    kernel_range = range(-int(size/2), int(size/2) + 1)
    x, y = np.meshgrid(kernel_range, kernel_range)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def apply_gaussian_blur(image, kernel):
    """ Apply Gaussian blur to an image using a kernel. """
    # Convert image to array
    array = np.array(image)
    # Pad the array
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    array_padded = np.pad(array, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    # Create an empty array to store the blurred image
    blurred = np.zeros_like(array)
    
    # Convolve the kernel over the image
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            blurred[i, j] = np.sum(array_padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    
    return Image.fromarray(blurred)

def sobel_filters(img):
    """Apply Sobel filters to an image to find gradients."""
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]])
    
    img_array = np.array(img)
    height, width = img_array.shape
    grad_mag = np.zeros_like(img_array)
    grad_dir = np.zeros_like(img_array, dtype=float)

    # Iterate over each pixel excluding the border
    for i in range(1, height-1):
        for j in range(1, width-1):
            region = img_array[i-1:i+2, j-1:j+2]
            px = np.sum(Gx * region)
            py = np.sum(Gy * region)
            
            grad_mag[i, j] = np.sqrt(px**2 + py**2)
            grad_dir[i, j] = np.arctan2(py, px)

    return grad_mag, grad_dir




def non_maximum_suppression(grad_mag, grad_dir):
    """Apply non-maximum suppression to thin the edges."""
    M, N = grad_mag.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = grad_dir * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                # Angle quantization
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = grad_mag[i, j+1]
                    r = grad_mag[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = grad_mag[i+1, j-1]
                    r = grad_mag[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = grad_mag[i+1, j]
                    r = grad_mag[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = grad_mag[i-1, j-1]
                    r = grad_mag[i+1, j+1]

                # Non-maximum suppression
                if (grad_mag[i,j] >= q) and (grad_mag[i,j] >= r):
                    Z[i,j] = grad_mag[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z
def double_threshold(nms_image, low_threshold, high_threshold):
    # Create binary images for strong and weak edges
    strong_edges = np.zeros_like(nms_image, dtype=np.uint8)
    weak_edges = np.zeros_like(nms_image, dtype=np.uint8)
    
    # Strong edges
    strong_edges[nms_image >= high_threshold] = 255
    
    # Weak edges
    weak_edges[(nms_image < high_threshold) & (nms_image >= low_threshold)] = 255
    
    return strong_edges, weak_edges
def hysteresis(strong_edges, weak_edges):
    height, width = strong_edges.shape

        # Initialize the final edges image
    final_edges = np.zeros((height, width), dtype=np.uint8)
        
        # Check if the weak edge pixels are connected to strong edge pixels
    for i in range(1, height-1):
        for j in range(1, width-1):
            if weak_edges[i, j] != 0:
                # Check if one of the neighbors is a strong edge
                if ((strong_edges[i+1, j-1] == 255) or (strong_edges[i+1, j] == 255) or
                    (strong_edges[i+1, j+1] == 255) or (strong_edges[i, j-1] == 255) or
                    (strong_edges[i, j+1] == 255) or (strong_edges[i-1, j-1] == 255) or
                    (strong_edges[i-1, j] == 255) or (strong_edges[i-1, j+1] == 255)):
                    final_edges[i, j] = 255
                else:
                    final_edges[i, j] = 0
            # Strong edges are always part of the final edge map
            elif strong_edges[i, j] == 255:
                final_edges[i, j] = 255

    return final_edges
def apply_canny(image_path, sigma, kernel_size, low_threshold, high_threshold):
    # Step 1: Load image and convert to grayscale
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian blur
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_img = apply_gaussian_blur(image_gray, kernel)
    
    # Step 3: Compute gradients using Sobel filters
    gradient_magnitude, gradient_direction = sobel_filters(blurred_img)
    
    # Step 4: Apply Non-maximum Suppression
    nms_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # Step 5: Apply Double Thresholding
    strong_edges, weak_edges = double_threshold(nms_image, low_threshold, high_threshold)
    
    # Step 6: Apply Hysteresis
    final_edges = hysteresis(strong_edges, weak_edges)

    return final_edges

# Example usage

