import time
import os
import shutil
import subprocess as proc
import numpy as np
from PIL import Image

OBS_TEMP = 'data/obs/temp/'
OBS_DATA = 'data/obs/data/'

def save_im(im):
    if (np.argmax(im) > 0):
        path = f'{OBS_TEMP}{time.time()}.jpg'
        img = Image.fromarray(im)
        img.convert('RGB').save(path)

def load_im():
    files = os.listdir(OBS_DATA)
    if (len(files) > 0):
        img = Image.open(f'{OBS_DATA}{files[0]}').convert('LA')
        im = np.asarray(img, dtype=np.float32)
        return im
    else:
        return None

def clean():
    proc.call(['image-cleaner', OBS_TEMP])

def move_temp_to_data():
    for fname in OBS_TEMP:
        fname = os.path.join(OBS_TEMP, fname)
        if os.path.isfile(fname):
            shutil.copy(fname, OBS_DATA)
            os.remove(fname)
