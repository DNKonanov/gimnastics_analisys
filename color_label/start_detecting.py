import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import argparse
from colors_processing import detect_segment, parse_colors_config
import os
import sys
import warnings
import subprocess

if not sys.warnoptions:
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser()

parser.add_argument('-video', type=str, help='path to video file', required=True)
parser.add_argument('-config', type=str, default='colors_config', help='colors config file')
parser.add_argument('-outdir', type=str, default='output', help='folder to store labeled frames from video')
parser.add_argument('-framerate', type=int, default=25, help='recording video framerate')
parser.add_argument('-square', type=int, default=250, help='the minimal square of detected segment')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)


try:
    os.mkdir(args.outdir)
except:
    pass

Colors = parse_colors_config(args.config)

print('Segments detection...')
i = 0
while True:

    print('Frame {}'.format(i), end='')
    ret, img = cap.read()

    if ret == False:
        break

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    

    seg = []
    for color in Colors:
        
        lower = np.array(Colors[color][0])
        upper = np.array(Colors[color][1])
        mask = cv2.inRange(hsv, lower, upper)
        
        seg += detect_segment(mask)

    fig, ax = plt.subplots(1, figsize=(14,7))
    
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for s in seg:


        if (s[1] - s[0])*(s[3] - s[2]) < args.square:
            continue
        rect = Rectangle((s[0], s[2]), s[1] - s[0], s[3] - s[2], 
                        linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    
    plt.savefig('{}/{}.png'.format(args.outdir, i), format='png')
    plt.close()

    i += 1
    print('\r', end='')
print()
print('Video recording...')
subprocess.call('ffmpeg -framerate {} -i {}/%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {}/segmented_video.mp4'.format(args.framerate, args.outdir, args.outdir), shell=True)

print('Completed!')