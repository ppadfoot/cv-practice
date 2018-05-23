import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.transform import rescale
from math import ceil, floor

def crop(img, percentage=6):
    height, width = img.shape
    beg_h, end_h = ceil((height * percentage) / 100), ceil(height - (height * percentage) / 100)
    beg_w, end_w = ceil((width * percentage) / 100), ceil(width - (width * percentage) / 100)
    return img[beg_h:end_h, beg_w:end_w], beg_h

def find_offset_one_dim(first_color, second_color, error_func, x_range, y_range):
    first_color, _ = crop(first_color, 20)
    second_color, _ = crop(second_color, 20)
    min_error = error_func(first_color, second_color)
    x_min = 0
    y_min = 0
    for x in x_range:
        for y in y_range:
            temp_first = np.roll(first_color, x, axis=0)
            temp_first = np.roll(temp_first, y, axis=1)
            temp_second = second_color
            new_error = error_func(temp_first, temp_second)
            if (new_error < min_error):
                min_error = new_error
                x_min = x
                y_min = y
    return (x_min, y_min)

def find_offset(first_img, second_img):
    if (first_img.shape[0] < 300):
        return find_offset_one_dim(first_img, second_img, 
                                   mean_squared_error, 
                                   np.arange(-15, 16), 
                                   np.arange(-15, 16))


    temp_first = rescale(first_img, .5, mode='constant')
    temp_second = rescale(second_img, .5, mode='constant')
    
    dx, dy = find_offset(temp_first, temp_second)
    dx *= 2
    dy *= 2
    return find_offset_one_dim(first_img, second_img, 
                               mean_squared_error, 
                               np.arange(dx - 1, dx + 2), 
                               np.arange(dy - 1, dy + 2))

def align(bgr_image, g_coord):
    b_row = b_col = r_row = r_col = 0
    (b, _), (g, _), (r, _) = map(crop, np.split(bgr_image[:bgr_image.shape[0] // 3 * 3], 3))
    avg_height = b.shape[0]

    dx_blue, dy_blue = find_offset(g, b)
    dx_red, dy_red = find_offset(g, r)

    b_row = g_coord[0] + dx_blue - bgr_image.shape[0] // 3 
    b_col = g_coord[1] + dy_blue

    r_row = g_coord[0] + dx_red + bgr_image.shape[0] // 3
    r_col = g_coord[1] + dy_red
        
    return bgr_image, (b_row, b_col), (r_row, r_col)

