#!/bin/sh
#Per Sahba's comments on Piazza, a 
# script to convert all .mp3 to .wav
files=$(ls *.mp3)
for x in $files
do
    target="${x%.*}"
    ffmpeg -i $x "${target}.wav"
done
