import numpy as np
from pyqtree import Index
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import io
from fractal_compression import compress
from PIL import Image

import sys
sys.path.append('./')

from utils import load_im

max_depth = 6
bias = 0.0125

a = 1.0 # visual_complexity_weight
b = 0.4 # processing_complexity_left_weight
c = 0.2 # processing_complexity_right_weight

class Region:
    def __init__(self, location, size, level):
        self.location = location
        self.size = size
        self.level = level

def split4(im):
    top_bottom = np.split(im, 2, axis=0)
    top_lr = np.split(top_bottom[0], 2, axis=1)
    bottom_lr = np.split(top_bottom[1], 2, axis=1)
    return top_lr[0], top_lr[1], bottom_lr[0], bottom_lr[1]

def concat4(tl, tr, bl, br):
    top = np.concatenate((tl, tr), axis=1)
    bottom = np.concatenate((tl, tr), axis=1)
    return np.concatenate((top, bottom), axis=0)

def mean4(splits):
    return np.mean(splits[0], axis=(0,1)), np.mean(splits[1], axis=(0,1)), np.mean(splits[2], axis=(0,1)), np.mean(splits[3], axis=(0,1))

def max4(splits):
    return np.max(splits[0], axis=(0,1)), np.max(splits[1], axis=(0,1)), np.max(splits[2], axis=(0,1)), np.max(splits[3], axis=(0,1))

def equal4(measurements):
    first = measurements[0]
    return all((x == first).all() for x in measurements)

def traverse(im, bbox, depth, qtree):
    if (depth > max_depth): 
        return

    splits = split4(im)
    w, h = splits[0].shape
    measurements = mean4(splits)
    mean = np.mean(measurements)

    if (mean > bias and mean < (1.0-bias)):
        for i in range(0, 4):
            ix, iy = i%2, i//2
            x1, y1 = bbox[0]+ix*w, bbox[1]+iy*h
            x2, y2 = x1+w, y1+h
            _bbox = (x1, y1, x2, y2)

            qtree.insert(Region(location=(x1, y1), size=(w, h), level=depth), _bbox)
            traverse(splits[i], _bbox, depth+1, qtree)

    return qtree

def evaluate(im, save=False):
    width, height = im.shape
    rect_im = (0, 0, width, height)

    # fractal method
    qtree = traverse(im, rect_im, 1, Index(bbox=rect_im))
    regions = qtree.intersect(rect_im)

    transformations = compress(im, 8, 4, 8)

    # rmse: np.sqrt(np.mean(np.square(target - img)))
    PCt0 = sum(1 for r in regions if r.level < max_depth-1)
    PCt1 = sum(1 for r in regions if r.level < max_depth)
    PC = len(regions)/im.size # processing_compression_ratio

    # jpeg method
    im_pil = Image.fromarray(im*255)
    im_pil = im_pil.convert('RGB')
    jpegbuf = io.BytesIO()
    im_pil.save(jpegbuf, format='JPEG', quality=95)

    nbytes_rgb_raw = im.size
    nbytes_jpeg = jpegbuf.getbuffer().nbytes
    IC = nbytes_jpeg/nbytes_rgb_raw # complexity compression ratio

    # fitness
    fitness = IC**a / (PCt0*PCt1)**b * ((PCt1-PCt0)/PCt1)**c
    #fitness = IC / PC

    if save:
        fig, ax = plt.subplots(1)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.imshow(im)

        for r in regions:
            x1, y1 = r.location
            w, h = r.size
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.text(0, 1, f'{round(fitness, 8)}', fontsize=12, color='w')
        print(f'P compression ratio: {PC}\nC compression ratio: {IC}')
        print(f'{nbytes_rgb_raw} raw bytes\n{nbytes_jpeg} compressed bytes\n{len(regions)} regions / {im.size} px')

        plt.savefig(f'results/eval/{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

    return fitness


img = load_im('results/eval/fur256.png')
evaluate(img, True)

# img = load_im('results/eval/0_test1.jpg')
# evaluate(img, True)

# img = load_im('results/eval/0_test2.jpg')
# evaluate(img, True)

# img = load_im('results/eval/0_test3.jpg')
# evaluate(img, True)

# img = load_im('results/eval/circle.png')
# evaluate(img, True)

# img = load_im('results/eval/noise.png')
# evaluate(img, True)


print('done!')
