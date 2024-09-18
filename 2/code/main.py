import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from skimage import io as skio
import cv2


#Part 1.1 Finite Difference Operator

def finite_difference_operator(image_path):
    im = skio.imread(image_path, as_gray=True)

    D_x = np.array([[1, -1]])
    D_y = np.array([[1], [-1]])

    grad_x = signal.convolve2d(im, D_x, boundary='symm', mode='same')
    grad_y = signal.convolve2d(im, D_y, boundary='symm', mode='same')

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y**2)

    threshold = 0.25
    edge_image = grad_magnitude > threshold 

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(im, cmap='gray')
    axes[0].set_title('Original Image')

    axes[1].imshow(np.abs(grad_x), cmap='gray')
    axes[1].set_title('Gradient in X')

    axes[2].imshow(np.abs(grad_y), cmap='gray')
    axes[2].set_title('Gradient in Y')

    axes[3].imshow(edge_image, cmap='gray')
    axes[3].set_title('Edge Image')

    for ax in axes:
        ax.axis('off')

    plt.show()

finite_difference_operator('data/cameraman.png')

# Part 1.2 Derivative of Gaussian Filter
def create_gaussian_kernel(ksize=5, sigma=1):
    g_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_2d = np.outer(g_1d, g_1d.T)
    return gaussian_2d

def smooth_then_derivative(image_path, ksize=5, sigma=1):
    im = skio.imread(image_path, as_gray=True)

    gaussian_kernel = create_gaussian_kernel(ksize, sigma)
    blurred_image = signal.convolve2d(im, gaussian_kernel, boundary='symm', mode='same')

    D_x = np.array([[1, -1]])
    D_y = np.array([[1], [-1]])  

    grad_x = signal.convolve2d(blurred_image, D_x, boundary='symm', mode='same')
    grad_y = signal.convolve2d(blurred_image, D_y, boundary='symm', mode='same')

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    threshold = 0.15
    edge_image = grad_magnitude > threshold

    return im, blurred_image, grad_x, grad_y, grad_magnitude, edge_image

def show_images(image_path, ksize=5, sigma=1):
    im, blurred_image, grad_x, grad_y, grad_magnitude, edge_image = smooth_then_derivative(image_path, ksize, sigma)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(im, cmap='gray')
    
    plt.subplot(2, 3, 2)
    plt.title('Blurred Image')
    plt.imshow(blurred_image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Gradient X')
    plt.imshow(grad_x, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title('Gradient Y')
    plt.imshow(grad_y, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title('Gradient Magnitude')
    plt.imshow(grad_magnitude, cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title('Edge Image')
    plt.imshow(edge_image, cmap='gray')

    plt.tight_layout()
    plt.show()


show_images('data/cameraman.png', ksize=5, sigma=1)

def derivative_of_gaussian(ksize=5, sigma=1):
    g_1d = cv2.getGaussianKernel(ksize, sigma)
    g_deriv_1d = cv2.getDerivKernels(1, 0, ksize, normalize=True)[0].reshape(-1, 1)
    gaussian_deriv_x = np.outer(g_deriv_1d, g_1d.T)
    gaussian_deriv_y = np.outer(g_1d, g_deriv_1d.T)

    return gaussian_deriv_x, gaussian_deriv_y

def convolve_with_derivative_of_gaussian(image_path, ksize=5, sigma=1):
    im = skio.imread(image_path, as_gray=True)

    gaussian_deriv_x, gaussian_deriv_y = derivative_of_gaussian(ksize, sigma)

    grad_x = signal.convolve2d(im, gaussian_deriv_x, boundary='symm', mode='same')
    grad_y = signal.convolve2d(im, gaussian_deriv_y, boundary='symm', mode='same')

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    threshold = 0.15
    edge_image = grad_magnitude > threshold

    return im, grad_x, grad_y, grad_magnitude, edge_image

def show_derivative_of_gaussian(image_path, ksize=5, sigma=1):
    im, grad_x, grad_y, grad_magnitude, edge_image = convolve_with_derivative_of_gaussian(image_path, ksize, sigma)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(im, cmap='gray')
    
    plt.subplot(2, 3, 2)
    plt.title('Gradient X (Derivative of Gaussian)')
    plt.imshow(grad_x, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Gradient Y (Derivative of Gaussian)')
    plt.imshow(grad_y, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title('Gradient Magnitude')
    plt.imshow(grad_magnitude, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title('Edge Image')
    plt.imshow(edge_image, cmap='gray')

    plt.tight_layout()
    plt.show()

show_derivative_of_gaussian('data/cameraman.png', ksize=5, sigma=1)
