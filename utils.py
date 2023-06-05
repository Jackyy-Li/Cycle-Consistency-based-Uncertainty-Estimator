import random
from skimage.util import random_noise
import matplotlib.pyplot as plt
import numpy as np


# noise function adds random noise to the input image with a random level
def noise(input_img, noise_level):
    # Generate random noise type and level
    noise_type = random.choice(['gaussian', 'salt', 'pepper', 's&p', 'speckle'])

    if noise_type == 'gaussian':
        noised_img = random_noise(input_img, mode='gaussian', var=noise_level)
    elif noise_type == 'salt':
        noised_img = random_noise(input_img, mode='salt', amount=noise_level)
    elif noise_type == 'pepper':
        noised_img = random_noise(input_img, mode='pepper', amount=noise_level)
    elif noise_type == 's&p':
        noised_img = random_noise(input_img, mode='s&p', amount=noise_level)
    elif noise_type == 'speckle':
        noised_img = random_noise(input_img, mode='speckle', var=noise_level)
    else:
        noised_img = input_img
        print("No noise added")

    return noised_img


# random_crop function crops the input image to 1/(crop_ratio^2) of its original size
def random_crop(image: np.ndarray, crop_ratio) -> np.ndarray:
    height, width = image.shape[:2]
    crop_height = int(height / crop_ratio)
    crop_width = int(width /  crop_ratio)

    y = np.random.randint(0, height - crop_height)
    x = np.random.randint(0, width - crop_width)
    return image[y:y+crop_height, x:x+crop_width]