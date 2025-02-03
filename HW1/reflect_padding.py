import numpy as np
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--laplacian', action='store_true')
    args = parser.parse_args()
    return args 

def reflect_padding(image, kernel_size):
    padded_image = cv2.copyMakeBorder(image, top=kernel_size, bottom=kernel_size, left=kernel_size, right=kernel_size, borderType=cv2.BORDER_REFLECT)
    return padded_image

def padding(input_img, kernel_size):
    height, width, channels = input_img.shape
    # Calculate the size based on the kernel size
    pad_size = kernel_size // 2 
    # Create a Padded image filled with zeros
    padded_image = np.zeros((height + 2 * pad_size, width + 2 * pad_size, 3))
    for c in range(channels):
        # Copy the original image into the center
        padded_image[pad_size:pad_size + height, pad_size:pad_size + width, c] = input_img[:, :, c]
    return padded_image

def convolution(input_img, kernel):
    kernel_size = kernel.shape[0]
    height, width, channels = input_img.shape
    convolve_img = np.zeros_like(input_img, dtype=np.float32)
    # padded_image = padding(input_img, kernel_size)
    padded_image = reflect_padding(input_img, kernel_size)

    # Convolution operation
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                # Extract the region of the padded image corresponding to the kernel
                region = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                # Compute the value for the current pixel
                convolve_img[i, j, c] = np.sum(region * kernel)
    convolve_img = np.clip(convolve_img, 0, 255).astype(np.uint8)
    return convolve_img
    
def gaussian_filter(input_img, kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    # Construct the Gaussian kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Distance from the center
            dx = i - center
            dy = j - center
            # Calculate the Gaussian at the current position
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(dx**2 + dy**2) / (2*sigma**2))
    # Normalize
    kernel = kernel / np.sum(kernel)
    # Apply convolution with the Gaussian kernel
    return convolution(input_img, kernel)

def median_filter(input_img, kernel_size):
    height, width, channels = input_img.shape
    padded_image = padding(input_img, kernel_size)
    output_img = np.zeros_like(input_img)
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                # Compute the median value of the neighborhood
                output_img[i, j, c] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size, c])
    return output_img

def laplacian_sharpening(input_img, idx):
    if idx == 1:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif idx == 2:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    laplacian_img = convolution(input_img, kernel)
    # Apply convolution with the Laplacian kernel
    return laplacian_img

if __name__ == "__main__":
    args = parse_args()

    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        print('Gaussian filter')
        kernel_size = int(input('Kernel size: '))
        sigma = float(input('Sigma: '))
        output_img = gaussian_filter(input_img, kernel_size, sigma)
        save_name = f'gaussian_kernel{kernel_size}_sigam{sigma}_reflect'
    elif args.median:
        print('Median filter')
        input_img = cv2.imread("input_part1.jpg")
        kernel_size = int(input('Kernel size: '))
        output_img = median_filter(input_img, kernel_size)
        save_name = f'median_filter_size{kernel_size}_reflect'
    elif args.laplacian:
        print('Laplacian filter')
        idx = int(input('Kernel 1 or 2: '))
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img, idx)
        save_name = f'laplacian_filter_kernrl{idx}_reflect'

    cv2.imwrite(f"{save_name}.jpg", output_img)