import time
import os
import numpy as np
from PIL import Image

# for naive data collecting
def save_im(im):
    if (np.argmax(im) > 0):
        path = f'data/canvas/{time.time()}.jpg'
        img = Image.fromarray(im)
        img.save(path)

def load_im():
    path = f'data/canvas/'
    files = os.listdir(path)
    if (len(files) > 0):
        img = Image.open(f'{path}{files[0]}')
        im = np.asarray(img, dtype=np.float32)
        return im
    else:
        return None
