# coding=utf-8
import numpy as np
from skimage.color import convert_colorspace
from datetime import datetime

def get_brightness(img):
     return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]


def get_energy(brightness, mask):
    img_energy = np.ones(brightness.shape)
    for x in range(brightness.shape[0]):
        for y in range(brightness.shape[1]):
            if x == 0:
                dx = brightness[1, y] - brightness[0, y]
            elif x == brightness.shape[0] - 1:
                dx = brightness[x, y] - brightness[x - 1, y]
            else:
                dx = brightness[x + 1, y] - brightness[x - 1, y]

            if y == 0:
                dy = brightness[x, 1] - brightness[x, 0]
            elif y == brightness.shape[1] - 1:
                dy = brightness[x, y] - brightness[x, y - 1]
            else:
                dy = brightness[x, y + 1] - brightness[x, y - 1]

            img_energy[x][y] = np.sqrt(dx**2 + dy**2)

    return img_energy + mask * brightness.shape[0] * brightness.shape[1] * 256


def get_seam_coords(img_energy):
    dp_array = np.copy(img_energy)
    for x in range(1, dp_array.shape[0]):
        for y in range(dp_array.shape[1]):
            if y == 0:
                dp_array[x][y] += min(dp_array[x - 1][y], dp_array[x - 1][y + 1])
            elif y == dp_array.shape[1] - 1:
                dp_array[x][y] += min(dp_array[x - 1][y], dp_array[x - 1][y - 1])
            else:
                dp_array[x][y] += min(dp_array[x - 1][y], dp_array[x - 1][y - 1], dp_array[x - 1][y + 1])

    min_energy = dp_array[-1, -1]
    ind = dp_array.shape[1] - 1
    for y in range(dp_array.shape[1]):
        if dp_array[-1, y] < min_energy:
            min_energy = dp_array[-1, y]
            ind = y


    x = dp_array.shape[0] - 1
    y = ind

    seam = [(x, y)]

    while x != 0:
        for shift in (-1, 0, 1):
            if 0 <= y + shift <= dp_array.shape[1] - 1:
                if dp_array[x][y] == dp_array[x - 1][y + shift] + img_energy[x][y]:
                    x -= 1
                    y += shift
                    seam.append((x, y))
                    break

    return tuple(reversed(seam))


def resize_image(img, seam, mode, mask):
    carve_mask = np.zeros(img.shape[0:2], dtype='uint8')
    if mode == 'shrink':
        resized_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), dtype='uint8')
        resized_mask = np.zeros((mask.shape[0], mask.shape[1] - 1), dtype='int8')
        for x in range(img.shape[0]):
            resized_img[x] = np.delete(img[x], seam[x][1], axis=0)
            carve_mask[seam[x]] = 1
            resized_mask[x] = np.delete(mask[x], seam[x][1])
    if mode == 'expand':
        resized_img = np.zeros((img.shape[0], img.shape[1] + 1, img.shape[2]), dtype='uint8')
        resized_mask = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype='int8')
        for x in range(img.shape[0]):
            if seam[x][1] == img.shape[1] - 1:
                new_pixel = (img[seam[x]] / 2 + img[seam[x][0], seam[x][1] - 1] / 2)
            else:
                new_pixel = (img[seam[x]] / 2 + img[seam[x][0], seam[x][1] + 1] / 2)
            carve_mask[seam[x]] = 1
            resized_img[x] = np.insert(img[x], seam[x][1] + 1, new_pixel, axis=0)
            resized_mask[x] = np.insert(mask[x], seam[x][1] + 1, 0)
            resized_mask[seam[x]] = 1

    return resized_img, carve_mask, resized_mask


def seam_carve(img, mode, mask=None):
    direction, mode = mode.split(' ')
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype='int64')
    else:
        mask = mask.astype('int64')
    if direction == 'vertical':
        img = np.swapaxes(img, 0, 1)
        mask = np.swapaxes(mask,0, 1)

    brightness = get_brightness(img)
    img_energy = get_energy(brightness, mask)
    seam = get_seam_coords(img_energy)
    resized_img, carve_mask, resized_mask = resize_image(img, seam, mode, mask)

    if direction == 'vertical':
        resized_img = np.swapaxes(resized_img, 0, 1)
        carve_mask = np.swapaxes(carve_mask, 0, 1)
        resized_mask = np.swapaxes(resized_mask, 0, 1)
    return (resized_img, resized_mask, carve_mask)
