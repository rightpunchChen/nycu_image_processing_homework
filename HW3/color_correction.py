import cv2
import numpy as np
import os


"""
TODO White patch algorithm
"""
def white_patch_algorithm(img):
    B, G, R = cv2.split(img.astype('float')) # Split the channel
    R_max, G_max, B_max = np.max(R), np.max(G), np.max(B) # Maximum of each channel
    n_R, n_G, n_B = 255 / R_max, 255 / G_max, 255 / B_max # Normalize
    R = np.clip(R * n_R, 0, 255) # Balance
    G = np.clip(G * n_G, 0, 255)
    B = np.clip(B * n_B, 0, 255)
    balanced_image = cv2.merge([B, G, R]).astype('uint8')
    return balanced_image



"""
TODO Gray-world algorithm
"""
def gray_world_algorithm(img):
    B, G, R = cv2.split(img.astype('float'))
    R_mean, G_mean, B_mean= np.mean(R), np.mean(G), np.mean(B) # Mean of each channel
    avg = (R_mean + G_mean + B_mean) / 3
    n_R, n_G, n_B = avg / R_mean, avg / G_mean, avg / B_mean # Normalize
    R = np.clip(R * n_R, 0, 255) # Balance
    G = np.clip(G * n_G, 0, 255)
    B = np.clip(B * n_B, 0, 255)
    balanced_image = cv2.merge([B, G, R]).astype('uint8')
    return balanced_image


"""
Bonus
"""
def other_white_balance_algorithm(img, p=6):
    """
    Reference:
    Finlayson, Graham D., et al.
    "Shades of gray and colour constancy."
    """
    B, G, R = cv2.split(img.astype('float')) # Split the channel
    R_norm = np.power(np.mean(np.power(R, p)), 1/p)
    G_norm = np.power(np.mean(np.power(G, p)), 1/p)
    B_norm = np.power(np.mean(np.power(B, p)), 1/p)
    avg = (R_norm + R_norm + R_norm) / 3
    n_R = avg / R_norm
    n_G = avg / G_norm
    n_B = avg / B_norm

    R = np.clip(R * n_R, 0, 255)
    G = np.clip(G * n_G, 0, 255)
    B = np.clip(B * n_B, 0, 255)
    balanced_image = cv2.merge([B, G, R]).astype('uint8')
    return balanced_image


"""
Main function
"""
def main():
    os.makedirs("result/color_correction", exist_ok=True)
    for i in range(2):
        img = cv2.imread("data/color_correction/input{}.png".format(i + 1))

        # TODO White-balance algorithm
        white_patch_img = white_patch_algorithm(img)
        gray_world_img = gray_world_algorithm(img)
        p = 6
        shades_of_gray_img = other_white_balance_algorithm(img, p=p)

        # cv2.imwrite("result/color_correction/white_patch_input{}.png".format(i + 1), white_patch_img)
        # cv2.imwrite("result/color_correction/gray_world_input{}.png".format(i + 1), gray_world_img)
        # cv2.imwrite("result/color_correction/shades_of_gray_input{}.png".format(i + 1), shades_of_gray_img)
        cv2.imshow('fig', np.hstack([img, white_patch_img, gray_world_img, shades_of_gray_img]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()