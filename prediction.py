"""
predict.py
Usage: python predict.py {Input Video} {(optional) destination folder}

Takes in an input video file and runs it through the entire process and prdiction pipeline

Video file must be 1920 x 1080 @ 60FPS
Video should contin gameplay of Counter Strike: Global Offensive
"""

import os
import sys
import random
import subprocess


# Ensure that there is an input file
assert len(sys.argv) >= 2, "Requires path to input file"
inFile = sys.argv[1]
assert os.path.isfile(inFile), f'{inFile} isn\'t a valid file path'

dir = "temp"
delDir = True

# allow a specified output directory
if len(sys.argv) > 2:
    dir = sys.argv[2]
    assert os.path.isdir(dir), f'{dir} isn\'t a valid dirctory'
    assert len(os.listdir(dir)) == 0, "folder must be empty"
    delDir = False

#  Make a new unique folder if none is specified
if delDir:
    while os.path.isdir(dir):
        dir = "temp" + str(random.randint(1,1000000))
    os.mkdir(dir)

# Run auto clip to get clips in folder
print(f'python auto_clip.py \'{inFile}\' \'{dir}\' 1')
subprocess.run(['python', './auto_clip.py', inFile, dir, '0'])

num_dirs = len([f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))])

# process clips from auto_clip with cnn to extract features
print('\nNow extracting features')
subprocess.run(['python', 'save_cnn_output.py', dir, dir])

# converts frames to grayscale
print('\nNow converting to grayscale')
for i in range(num_dirs):
    subprocess.run(['python', 'convert_gray.py', dir + "/" + str(i) ])

# analyze frames with optical flow
print('\nNow analyzing frames')
subprocess.run(['python', 'analyze.py', dir + "/clips.pt", dir])


