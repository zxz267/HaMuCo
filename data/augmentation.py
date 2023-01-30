import random
import cv2
import numpy as np

def occlusion_augmentation(img, patch_size):
    '''
    :param img:
    :param patch_size:
    :return:
    '''
    height, width, channel = img.shape
    row, column = height // patch_size, width // patch_size
    patch_size = int(patch_size)
    mask_row, mask_column = random.randint(0, row-1), random.randint(0, column-1)
    img[mask_row*patch_size:(mask_row+1)*patch_size,
    mask_column*patch_size:(mask_column+1)*patch_size, :] \
        = np.zeros(shape=(patch_size, patch_size, 3))
    return img

def occlusionAugmentation(img, prob):
    if random.random() <= prob:
        patch_size = random.randint(8, 64)
        img = occlusion_augmentation(img, patch_size)
    return img

def blurAugmentation(img, prob):
    if random.random() <= prob:
        type = random.randint(1, 3)
        kernel_size = random.choice([3, 5, 7, 9, 11, 13])
        if type == 1:
            img = cv2.blur(img, (kernel_size, kernel_size))
        elif type == 2:
            img = cv2.medianBlur(img, kernel_size)
        elif type == 3:
            sigma = random.choice([0, 1, 2, 3, 4, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return img




