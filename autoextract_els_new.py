import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from gymnlib.extracting import parse_templates
from gymnlib.alignment import align_samples
from gymnlib.convolution import convolve, deconvolve
from gymnlib.distance import compute_distance
from fastdtw import fastdtw
from scipy.stats import spearmanr, linregress
from scipy.signal import find_peaks
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)


METRICS = ['LINREGRESS', 'SPEARMEN', 'INTEGRALDIFF']



parser = argparse.ArgumentParser()
parser.add_argument('-sample', type=str, default=None, help='n-dim timeline')
parser.add_argument('-templates', type=str, default='templates', help='path to a folder containing templates')
parser.add_argument('-step', type=int, default=5000, help='step of search (default 5000)')
parser.add_argument('-threshold', type=float, default=0.7, help='matching threshold (from 0 to 1, default 0.7)')
parser.add_argument('-outdir', type=str, default='out', help='out directory')
parser.add_argument('-metric', type=str, default='LINREGRESS', help='metric to calculate distance ("LINREGRESS", "SPEARMEN", "INTEGRALDIFF"')
parser.add_argument('-conv_val', type=int, default=100, help='convolution parameter')
parser.add_argument('-radius', type=int, default=10, help='size of neighborhood when expanding the path in the FastDTW method')
parser.add_argument('-use_distance', type=str, default=0, help='use distance flag (0 or 1, default 0)')
args = parser.parse_args()

if args.sample == None:
    print('choose sample!')
    sys.exit()

if args.metric not in METRICS:
    print('Invalid metric!')
    sys.exit()

Templates, Templates_sizes, Templates_coorections = parse_templates(args.templates, conv_val=args.conv_val)

Sample = pd.read_csv(args.sample, sep='\t', skiprows=3, header=None, names=[str(i) for i in range(17)], engine='python')
Sample = np.array(Sample.drop('16',axis=1).values)

Sample1 = convolve(Sample, conv_val=args.conv_val)

templates_matches = []

for key in Templates:

    t = Templates[key]

    templates_matches.append([])
    const_step = Templates_sizes[key]
    correction = Templates_coorections[key]

    
    diff = int(const_step/args.conv_val)
    
    for i in range(0, len(Sample1[0]) - diff, int(args.step/args.conv_val)):


        print('step {}'.format(int(i*args.conv_val/args.step)))
        print('{} to {} fragment...'.format(i*args.conv_val, (i + diff)*args.conv_val))
        

        sample = np.array([[Sample1.T[q][j] for j in range(len(Sample1.T[q]))] for q in range (i, i + diff)]).T


        listSample1, listTemplate, distance = align_samples(sample, t, radius=args.radius, al_type='REPLACE')

        if args.use_distance == 0:
            distance = 1
        else:
            distance = distance/len(listSample1)


        templates_matches[-1].append(compute_distance(listSample1, listTemplate, metric=args.metric, correction=correction, distance=distance))
    templates_matches[-1] = np.array(templates_matches[-1])
    print()

max_match = max([max(t) for t in templates_matches])

if distance != 1:
    for t in range(len(templates_matches)):
        templates_matches[t] = templates_matches[t]/max_match

dists = []

plt.figure(figsize=(20, 10))
for t in templates_matches:
    dists.append(np.array(t))

    plt.plot([d*args.step for d in range(len(dists[-1]))], dists[-1])

try:
    os.mkdir(args.outdir)
except FileExistsError:
    pass

templates_coords = [[] for d in dists]
for dist in range(len(dists)):

    norm = int(Templates_sizes[dist]*(20000/const_step)/(args.step*4))

    f_out = open(args.outdir + '/template_{}.txt'.format(dist+1), 'w')

    for point in range(len(dists[dist])):
        if dists[dist][point] <= args.threshold:
            continue

        skip = 0
        for p in range(max(0, point - norm), min(point + norm + 1, len(dists[dist]))):
            for d in range(len(dists)):
                try:
                    if dists[d][p] > dists[dist][point]:
                        skip = 1 
                        break
                except IndexError:
                    continue
        if skip == 0:
            templates_coords[dist].append((point*args.step, point*args.step+Templates_sizes[dist]))
            f_out.write('{} '.format(dists[dist][point]))


colors = ['r', 'g', 'b']

current_c = 0
t_num = 0

for t in templates_coords:
    for coord in t:
        plt.axvline(x=coord[0]-(Templates_sizes[t_num]/2), c=colors[current_c])
        plt.axvline(x=coord[1]-(Templates_sizes[t_num]/2), c=colors[current_c])
    
    current_c += 1
    t_num += 1


plt.savefig(args.outdir + '/templates_matching.png', dpi=300, format='png')
plt.show()
print(templates_coords)

Sample1 = deconvolve(Sample1)
current_template = 1
for t in templates_coords:
    for coord in t:

        Image = [[col[j] for j in range(coord[0], coord[1])] for col in Sample1]
        np.savetxt(args.outdir + '/template_{}_{}:{}.txt'.format(current_template, coord[0], coord[1]), Image)
        Image = [[col[j] for j in range(coord[0], coord[1], 200)] for col in Sample1]
        
        plt.imshow(Image)
        plt.savefig(args.outdir + '/template_{}_{}:{}.png'.format(current_template, coord[0], coord[1]), dpi=300, format='png')

    current_template += 1



    
