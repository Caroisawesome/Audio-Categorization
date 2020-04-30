# -*- coding: utf-8 -*-
"""
CS 529- Project 3: Audio Categorization

@author: Alyshia Bustos
"""

#import pandas as pd
import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

'''
===============================================================================
Start of our program
===============================================================================
'''
if __name__ == '__main__':

    #filename = "C:\Users\alysh\Documents\03-School\UNM_Class_Work\Spring2020\CS529\p3_audiocategorization\data\train/00907299.mp3'"
    data_dir = './data/project_waves_norm/train'
    audio_files = glob(data_dir + '/*.wav')

    print(len(audio_files))
    total_audios = len(audio_files)
    
    
    for file in range(total_audios):
    
        filename = audio_files[file].replace(data_dir + '\\' , '').replace('.wav','.png')
        audio, sfreq = lr.load(audio_files[file])
        time = np.arange(0,len(audio)) / sfreq
        fig, ax = plt.subplots()
        ax.plot(time, audio)
        plt.axis('off')
        plt.savefig('./project_time_series/train/'+ filename,bbox_inches='tight',transparent=True, pad_inches=0 )
