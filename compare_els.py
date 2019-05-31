from scipy.stats import spearmanr, linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sample1', type=str, default=None, help='raw file with angles data 1')
parser.add_argument('-sample2', type=str, default=None, help='raw file with abgles data 2')
parser.add_argument('-coord1', type=str, default=None, help='coords from sample 1 in format "leftcoord:rightcoord"')
parser.add_argument('-coord2', type=str, default=None, help='coords from sample 2 in format "leftcoord:rightcoord"')
parser.add_argument('-radius', type=int, default=1, help='size of neighborhood when expanding the path in the FastDTW method')

args = parser.parse_args()

if None in (args.sample1, args.sample2, args.coord1, args.coord2):
    print('Check parameters!')


else:

    Sample1 = pd.read_csv(args.sample1, sep='\t', skiprows=3, header=None, names=[str(i) for i in range(17)], engine='python')
    Sample2 = pd.read_csv(args.sample2, sep='\t', skiprows=3, header=None, names=[str(i) for i in range(17)], engine='python')


    l1, r1 = args.coord1.split(':')
    l2, r2 = args.coord2.split(':')
    l1, r1, l2, r2 = int(l1), int(r1), int(l2), int(r2)

    align_shifts = []

    for col in Sample1:
        if col == '16':
            continue
        
        S1 = list(Sample1[col][l1:r1])
        S2 = list(Sample2[col][l2:r2])
        
        
        align = fastdtw(S1, S2, radius=args.radius)

        el1_al = [S1[i[0]] for i in align[1]]
        el2_al = [S2[i[1]] for i in align[1]]
        
        align_shifts.append([i for i in align[1]])
        


    min_shift = min([len(i) for i in align_shifts])

    mean_shift1 = [np.median([j[i][0] for j in align_shifts]) for i in range(min_shift)]
    mean_shift2 = [np.median([j[i][1] for j in align_shifts]) for i in range(min_shift)]


    listSmple1 = {col: list(Sample1[col][l1:r1]) for col in Sample1 if col != '16'}
    listSmple2 = {col: list(Sample2[col][l2:r2]) for col in Sample2 if col != '16'}

    new_image1 = [[listSmple1[col][int(mean_shift1[i])] for i in range(0, len(mean_shift1), 200)] for col in listSmple1]
    new_image2 = [[listSmple2[col][int(mean_shift2[i])] for i in range(0, len(mean_shift2), 200)] for col in listSmple2]

    fig, axs = plt.subplots(3,1,figsize=(30,30))


    axs[0].set_title('{}, {}'.format(args.sample1, args.coord1))
    axs[0].imshow(new_image1)
    axs[1].set_title('{}, {}'.format(args.sample2, args.coord2))
    axs[1].imshow(new_image2)
    axs[2].set_title('Difference')
    axs[2].imshow(np.abs(np.array(new_image1) - np.array(new_image2)), cmap='GnBu', vmax=180)
    axs[2].text(2,2,'Spearman: {}, {}'.format((*spearmanr(np.array(new_image1).flatten(), np.array(new_image2).flatten()))), fontsize=12)
    axs[2].text(2,5,'Linregress: {}, {}'.format(*(linregress(np.array(new_image1).flatten(), np.array(new_image2).flatten()))[2:4]), fontsize=12)
    plt.show()

    print('Spearman: {}, {}'.format((*spearmanr(np.array(new_image1).flatten(), np.array(new_image2).flatten()))))
    print('Linregress: {}, {}'.format(*(linregress(np.array(new_image1).flatten(), np.array(new_image2).flatten()))[2:4]))


