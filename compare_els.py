from scipy.stats import spearmanr, linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-sample1', type=str, default=None, help='raw file with angles data 1')
parser.add_argument('-sample2', type=str, default=None, help='raw file with abgles data 2')
parser.add_argument('-coord1', type=str, default=None, help='coords from sample 1 in format "leftcoord:rightcoord"')
parser.add_argument('-coord2', type=str, default=None, help='coords from sample 2 in format "leftcoord:rightcoord"')
parser.add_argument('-radius', type=int, default=1, help='size of neighborhood when expanding the path in the FastDTW method')

args = parser.parse_args()

if None in (args.sample1, args.sample2, args.coord1, args.coord2):
    print('Check parameters!')
    sys.exit()




Sample1pd = pd.read_csv(args.sample1, sep='\t', skiprows=3, header=None, names=[str(i) for i in range(17)], engine='python')
Sample2pd = pd.read_csv(args.sample2, sep='\t', skiprows=3, header=None, names=[str(i) for i in range(17)], engine='python')

l1, r1 = args.coord1.split(':')
l2, r2 = args.coord2.split(':')
l1, r1, l2, r2 = int(l1), int(r1), int(l2), int(r2)


Sample1 = np.array([[Sample1pd[col][i] for i in range(l1, r1)] for col in Sample1pd if col != '16'])
Sample2 = np.array([[Sample2pd[col][i] for i in range(l2, r2)] for col in Sample2pd if col != '16'])

path = fastdtw(Sample1.T, Sample2.T, radius=args.radius)[1]

new_image1 = []
new_image2 = []

for p in path:
    new_image1.append(Sample1.T[p[0]])
    new_image2.append(Sample2.T[p[1]])

new_image1 = np.array(new_image1).T
new_image2 = np.array(new_image2).T

cropped_image1 = [[i[j] for j in range(0, len(i), 200)] for i in new_image1]
cropped_image2 = [[i[j] for j in range(0, len(i), 200)] for i in new_image2]

fig, axs = plt.subplots(3,1,figsize=(30,30))


axs[0].set_title('{}, {}'.format(args.sample1, args.coord1))
axs[0].imshow(cropped_image1)
axs[1].set_title('{}, {}'.format(args.sample2, args.coord2))
axs[1].imshow(cropped_image2)
axs[2].set_title('Difference')
axs[2].imshow(np.abs(np.array(cropped_image1) - np.array(cropped_image2)), cmap='GnBu', vmax=180)
axs[2].text(2,2,'Spearman: {}, {}'.format((*spearmanr(np.array(new_image1).flatten(), np.array(new_image2).flatten()))), fontsize=12)
axs[2].text(2,5,'Linregress: {}, {}'.format(*(linregress(np.array(new_image1).flatten(), np.array(new_image2).flatten()))[2:4]), fontsize=12)
plt.show()

print('Spearman: {}, {}'.format((*spearmanr(np.array(new_image1).flatten(), np.array(new_image2).flatten()))))
print('Linregress: {}, {}'.format(*(linregress(np.array(new_image1).flatten(), np.array(new_image2).flatten()))[2:4]))


