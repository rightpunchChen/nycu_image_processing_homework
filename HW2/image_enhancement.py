import cv2
import numpy as np


"""
TODO Part 1: Gamma correction
"""
def gamma_correction(img, gamma):
    # Normalize the pixel values ​​to [0, 1], 
    # apply gamma correction, and scale back to [0, 255]
    gamma_img = ((img / 255) ** gamma) * 255
    return gamma_img.astype(np.uint8)


"""
TODO Part 2: Histogram equalization
"""
def histogram_equalization(img):
    h, w = img.shape[:2]
    pixel_num = h * w
    hist = np.zeros((256, 3), dtype=np.float64)
    hist_sum = np.zeros((256, 3), dtype=np.float64)
    # Construct the histogramof each channel
    for c in range(3):
        for i in range(h):
            for j in range(w):
                hist[img[i,j,c],c] += 1
    # Calculate the cumulative distribution function (CDF) of each channel
    hist_sum[0,:] = hist[0,:]       
    for i in range(1, 256):
        hist_sum[i,:] = hist[i,:] + hist_sum[i-1,:]
    # Normalized the CDF to [0, 255]
    hist_sum = (np.round((hist_sum / pixel_num) * 255)).astype(np.uint8)
    # Assign new pixels of each channel
    he_img = np.zeros_like(img)
    for c in range(3):
        he_img[:,:,c] = hist_sum[img[:,:,c],c]
    return he_img

def histogram_equalization_new(img):
    h, w = img.shape[:2]
    pixel_num = h * w
    hist = np.zeros((256, 3), dtype=np.float64)
    hist_sum = np.zeros((256, 3), dtype=np.float64)
    for c in range(3):
        for i in range(h):
            for j in range(w):
                hist[img[i,j,c],c] += 1
    hist_sum[0,:] = hist[0,:]       
    for i in range(1, 256):
        hist_sum[i,:] = hist[i,:] + hist_sum[i-1,:]

    # Consider the minimum value of CDF when calculating normalization to avoid shifts in dark areas
    # This calculation method will make the output closer to the contrast enhancement effect of the human eye.
    for c in range(3):
        cdf_min = hist_sum[:,c].min()
        hist_sum[:,c] = ((hist_sum[:,c] - cdf_min) / (pixel_num - cdf_min) * 255).clip(0, 255)

    he_img = np.zeros_like(img)
    for c in range(3):
        he_img[:,:,c] = hist_sum[img[:,:,c],c]
    return he_img

def histogram_equalization_grayscale(img):
    h, w = img.shape
    pixel_num = h * w
    hist = np.zeros((256), dtype=np.float64)
    hist_sum = np.zeros((256), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]] += 1
    hist_sum[0] = hist[0]       
    for i in range(1, 256):
        hist_sum[i] = hist[i] + hist_sum[i-1]

    cdf_min = hist_sum.min()
    # hist_sum = (np.round((hist_sum / pixel_num) * 255)).astype(np.uint8)
    hist_sum = ((hist_sum - cdf_min) / (pixel_num - cdf_min) * 255).clip(0, 255)

    he_img = np.zeros_like(img)
    he_img[:,:] = hist_sum[img[:,:]]
    return he_img

"""
Bonus
"""
def other_enhancement_algorithm(img):
    from homomorphic_filter import homomorpgic_filter
    return homomorpgic_filter(img=img)


"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # TODO: modify the hyperparameter
    gamma_list = [2, 0.5, 0.25] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img.astype(np.float64), gamma)
        gamma_correction_g_img = gamma_correction(g_img.astype(np.float64), gamma)
        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.hstack([img, gamma_correction_img]))
        cv2.waitKey(0)
        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.hstack([g_img, gamma_correction_g_img]))
        cv2.waitKey(0)

    # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization_new(img)
    histogram_equalization_g_img = histogram_equalization_grayscale(g_img)
    cv2.imshow("Histogram equalization", np.hstack([img, histogram_equalization_img]))
    cv2.waitKey(0)
    cv2.imshow("Histogram equalization", np.hstack([g_img, histogram_equalization_g_img]))
    cv2.waitKey(0)

    homomorpgic_img = other_enhancement_algorithm(img)
    cv2.imshow("Homomorpgic filter", np.hstack([img, homomorpgic_img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
