import numpy as np

from os import path
from pydub import AudioSegment
import os

def make_dirs():
    try:
        os.makedirs("data/project_waves")
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs("data/project_waves/train")
    except FileExistsError:
        # directory already exists
        pass

def convert_mp3_to_wav(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

if (__name__ == '__main__'):


    in_path = 'data/project3/train/'
    out_path ='data/project_waves/train/'

    make_dirs()

    for root, dirs, files in os.walk(in_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            out  = os.path.join(out_path, name.split(".")[0]+".wav")

            convert_mp3_to_wav(path, out)

        # for name in dirs:
        #   print(os.path.join(root, name))

