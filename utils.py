import time
import os
import shutil
import torch
import subprocess as proc
import numpy as np

from PIL import Image
from es import compute_ranks

OBS_TEMP = 'data/obs/temp/'
OBS_DATA = 'data/obs/data/'

def save_im(im):
    if (np.max(im) > 0):
        path = f'{OBS_TEMP}{time.time()}.png'
        img = Image.fromarray(im)
        img.convert('RGB').save(path)

def load_im(path=None, id=0, normalize=True):
    if not path:
        files = os.listdir(OBS_DATA)
        if (len(files) > 0):
            _path = f'{OBS_DATA}{files[id]}'
        else: return None
    else:
        _path = path
    
    img = Image.open(_path).convert('L')
    im = np.asarray(img, dtype=np.float32)
    return im/255.0 if normalize else im

def clean():
    proc.call(['image-cleaner', OBS_TEMP])

def move_temp_to_data():
    for fname in OBS_TEMP:
        fname = os.path.join(OBS_TEMP, fname)
        if os.path.isfile(fname):
            shutil.copy(fname, OBS_DATA)
            os.remove(fname)

""" Save a checkpoint of the models """
def save_checkpoint(model, filename, state, current_time):

    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'saved_models', current_time)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = os.path.join(dir_path, "{}-{}.pth.tar".format(state['version'], filename))
    state['model'] = model.state_dict()
    torch.save(state, filename)
