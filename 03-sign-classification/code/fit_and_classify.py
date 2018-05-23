import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn import svm

def crop_and_rescale(im, size=(64,64)):
    im = rgb2gray(im)    
    dim = max(im.shape)
    return resize(im[:dim, :dim], size, mode='constant')

def make_histogram(mag, ang, cell_x, cell_y, pos_x, pos_y, bins_count):
    hist_vec = np.zeros((bins_count,))
    for y in range(pos_y, pos_y + cell_y):
        for x in range(pos_x, pos_x + cell_x):
            cur_ang = ang[y, x]
            ind = int(cur_ang // (180 / bins_count))
            cur_ang %= (180 / bins_count)
            hist_vec[(ind + 1) % bins_count] += (cur_ang / 20) * mag[y, x]
            hist_vec[ind % bins_count] += (1. - cur_ang / 20) * mag[y, x]
    return hist_vec

def extract_hog(im, bins_count=9, cell_size=(8, 8), block_size=(2, 2), eps=1e-25):
    im = crop_and_rescale(im)
    im = np.sqrt(im)
    dy, dx = np.gradient(im, .5)
    
    mag = np.hypot(dx, dy)
    ang = np.rad2deg(np.arctan2(dy, dx)) % 180
    
    cell_x, cell_y = cell_size
    block_x, block_y = block_size

    count_cx = im.shape[1] // cell_x
    count_cy = im.shape[0] // cell_y

    histograms = np.zeros((count_cy, count_cx, bins_count))

    for y in range(count_cy):
        for x in range(count_cx):
            histograms[y, x] = make_histogram(mag, ang, cell_x, cell_y, cell_x * x, cell_y * y, bins_count)


    count_bx = count_cx - block_x + 1
    count_by = count_cy - block_y + 1
    normalized = np.zeros((count_by, count_bx, block_y, block_x, 9))
    for y in range(count_by):
        for x in range(count_bx):
            block = histograms[y:y + block_y, x:x + block_x, :]
            normalized[y, x, :] = block / np.sqrt(np.sum(block ** 2) + eps)

    return normalized.ravel()

def fit_and_classify(train_featues, train_labels, test_features):
    clf = svm.SVC(kernel='rbf', C=150)
    clf.fit(train_featues, train_labels)
    return clf.predict(test_features)