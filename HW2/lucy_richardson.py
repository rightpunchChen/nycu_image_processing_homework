import cv2
import numpy as np

def generate_motion_blur_psf(size, length ,theta):
    psf = np.zeros(size, np.float64)
    center = (size[1]//2, size[0]//2)
    axes = (0, length//2)
    cv2.ellipse(psf, center, axes, theta, 0, 360, 1, -1)
    psf /= psf.sum()
    return psf

def lucy_richardson(blurred_img, psf, iterations=200):
    estimate_img = blurred_img.copy()
    for _ in range(iterations):
        for c in range(3):
            # f_k * h
            conv_estimate = cv2.filter2D(estimate_img[:,:,c], -1, psf)
            # g/(f_k * h)
            relative_blur = blurred_img[:,:,c] / (conv_estimate + 1e-6)
            # (g/(f_k * h)) * h^*
            correction_factor = cv2.filter2D(relative_blur, -1, psf.T)
            # f_k * ((g/(f_k * h)) * h^*)
            estimate_img[:,:,c] *= correction_factor
    estimate_img = np.clip(estimate_img, 0, 255)
    return estimate_img.astype(np.uint8)

if __name__ == "__main__":
    blurred_img = cv2.imread("data/image_restoration/testcase1/input_blurred.png")
    psf = generate_motion_blur_psf(blurred_img.shape[:2], 40, 45)
    lucy_img = lucy_richardson(blurred_img.astype(np.float64), psf)
    cv2.imshow("window", np.hstack([blurred_img, lucy_img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()