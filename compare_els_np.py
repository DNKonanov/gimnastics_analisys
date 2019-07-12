from scipy.stats import spearmanr, linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import argparse
import sys
from gymnlib.convolution import convolve

parser = argparse.ArgumentParser()
parser.add_argument('-sample1', type=str, default=None, help='raw file with angles data 1')
parser.add_argument('-sample2', type=str, default=None, help='raw file with abgles data 2')
parser.add_argument('-conv_val', type=int, default=100, help='convolution value')
parser.add_argument('-radius', type=int, default=1, help='size of neighborhood when expanding the path in the FastDTW method')
parser.add_argument('-showplots', type=int, default=1, help='show plots (0 or 1)')
args = parser.parse_args()

if None in (args.sample1, args.sample2):
    print('Check parameters!')
    sys.exit()


Sample1 = convolve(np.loadtxt(args.sample1).T, conv_val=args.conv_val)
Sample2 = convolve(np.loadtxt(args.sample2).T, conv_val=args.conv_val)

distance, path = fastdtw(Sample1.T, Sample2.T, radius=args.radius)

new_image1 = []
new_image2 = []

for p in path:
    new_image1.append(Sample1.T[p[0]])
    new_image2.append(Sample2.T[p[1]])

new_image1 = np.array(new_image1).T
new_image2 = np.array(new_image2).T

cropped_image1 = new_image1
cropped_image2 = new_image2

fig, axs = plt.subplots(3,1,figsize=(30,30))


if args.showplots == 1:
    axs[0].set_title('{}'.format(args.sample1))
    axs[0].imshow(cropped_image1)
    axs[1].set_title('{}'.format(args.sample2))
    axs[1].imshow(cropped_image2)
    axs[2].set_title('Difference')
    axs[2].imshow(np.abs(np.array(cropped_image1) - np.array(cropped_image2)), cmap='GnBu', vmax=180)
    axs[2].text(2,2,'Spearman: {}, {}'.format((*spearmanr(np.array(new_image1).flatten(), np.array(new_image2).flatten()))), fontsize=12)
    axs[2].text(2,5,'Linregress: {}, {}'.format(*(linregress(np.array(new_image1).flatten(), np.array(new_image2).flatten()))[2:4]), fontsize=12)
    axs[2].text(2,8,'Distance {}'.format(distance))
    plt.show()

print('Spearman: {}, {}'.format((*spearmanr(np.array(new_image1).flatten(), np.array(new_image2).flatten()))))
print('Linregress: {}, {}'.format(*(linregress(np.array(new_image1).flatten(), np.array(new_image2).flatten()))[2:4]))
print('Distance {}'.format(abs(np.sum(cropped_image1-cropped_image2))))
