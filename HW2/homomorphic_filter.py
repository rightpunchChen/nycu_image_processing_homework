import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def homomorpgic_filter(img, D0=80.0, gammaH=2.0, gammaL=0.5, c=2.0):
    img_normalized = img.astype(np.float32) / 255.0
    enhanced_img = np.zeros_like(img_normalized)
    # Construct homomorphic filter H(u, v)
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    u = np.arange(-crow, crow)
    v = np.arange(-ccol, ccol)
    V, U = np.meshgrid(v, u)
    D = U**2 + V**2
    H = (gammaH - gammaL) * (1 - np.exp(-c * (D / D0**2))) + gammaL
    for channel in range(3):
        # Log transformation
        log_img = np.log1p(img_normalized[:,:,channel])
        # FFT
        img_fft = fftshift(fft2(log_img))
        # Apply homomorphic filter
        filtered_img = (gammaH - gammaL) * (img_fft * H) + gammaL
        filtered_img = ifft2(ifftshift(filtered_img))
        enhanced_img[:,:,channel] = np.exp(np.real(filtered_img)) - 1
    enhanced_img = np.clip(enhanced_img*255, 0, 255)
    return enhanced_img.astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread('data/image_enhancement/input.bmp')
    enhanced_img = homomorpgic_filter(img)
    cv2.imshow("window", np.hstack([img, enhanced_img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()