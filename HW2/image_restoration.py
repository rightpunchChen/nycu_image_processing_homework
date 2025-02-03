import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

"""
TODO Part 1: Motion blur PSF generation
"""
def generate_motion_blur_psf(size, length ,theta):
    psf = np.zeros(size, np.float64)
    center = (size[1]//2, size[0]//2)
    axes = (0, length//2)
    cv2.ellipse(psf, center, axes, theta, 0, 360, 1, -1)
    psf /= psf.sum()
    return psf

"""
TODO Part 2: Wiener filtering
"""
def wiener_filtering(blurred_img, psf, K):
    restored_img = np.zeros(blurred_img.shape[:])
    
    # Calculate Wiener filter (1 / H * |H|^2 / (|H|^2 + K))
    psf_fft = fft2(psf,s=blurred_img.shape[:2])
    psf_abs_squared = np.abs(psf_fft) ** 2
    wiener_filter = np.conj(psf_fft) / (psf_abs_squared + K)

    for c in range(3):
        # Apply the filter in the frequency domain
        blurred_fft = fft2(blurred_img[:,:,c])
        restored_fft = blurred_fft * wiener_filter
        restored_img[:,:,c] = np.real(fftshift(ifft2(restored_fft)))
    restored_img = np.clip(restored_img, 0, 255)
    return restored_img.astype(np.uint8)

"""
TODO Part 3: Constrained least squares filtering
"""
def constrained_least_square_filtering(blurred_img, psf, gamma):
    restored_img = np.zeros(blurred_img.shape[:])
    laplacian = [[0,-1,0],[-1,4,-1],[0,-1,0]]
    laplacian_fft = fft2(laplacian, s=blurred_img.shape[:2])
    laplacian_abs_squared = np.abs(laplacian_fft) ** 2

    # Compute the CLS filter in the frequency domain
    psf_fft = fft2(psf,s=blurred_img.shape[:2])
    psf_abs_squared = np.abs(psf_fft) ** 2
    cls_filter = np.conj(psf_fft) / (psf_abs_squared + gamma * laplacian_abs_squared)
    
    for c in range(3):
        blurred_fft = fft2(blurred_img[:,:,c])
        restored_fft = blurred_fft * cls_filter
        restored_img[:, :, c] = np.real(fftshift(ifft2(restored_fft)))
    restored_img = np.clip(restored_img, 0, 255)

    return restored_img.astype(np.uint8)

"""
Bouns
"""
def other_restoration_algorithm(blurred_img, psf):
    from lucy_richardson import lucy_richardson
    return lucy_richardson(blurred_img, psf)


def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr

"""
Main function
"""
def main():
    for i in range(2):
        original_img = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        blurred_img = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))
        
        # TODO Part 1: Motion blur PSF generation
        psf = generate_motion_blur_psf(blurred_img.shape[:2], 40, 45)

        # TODO Part 2: Wiener filtering
        wiener_img = wiener_filtering(blurred_img.astype(np.float64), psf, 0.002)

        # # TODO Part 3: Constrained least squares filtering
        constrained_least_square_img = constrained_least_square_filtering(blurred_img.astype(np.float64), psf, 0.05)

        lucy_richardson_img = other_restoration_algorithm(blurred_img.astype(np.float64), psf)

        print("\n---------- Testcase {} ----------".format(i))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(original_img, wiener_img)))

        print("Method: Constrained least squares filtering")
        print("PSNR = {}\n".format(compute_PSNR(original_img, constrained_least_square_img)))

        print("Method: Lucy Richardson Method")
        print("PSNR = {}\n".format(compute_PSNR(original_img, lucy_richardson_img)))

        cv2.imshow("window", np.hstack([blurred_img, wiener_img, constrained_least_square_img, lucy_richardson_img]))
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
