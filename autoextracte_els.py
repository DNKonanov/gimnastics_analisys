import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from fastdtw import fastdtw
from scipy.stats import spearmanr, linregress
from scipy.signal import find_peaks
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)



methods = ['R2', 'SPEARMEN', 'INTEGRALDIFF']



def compute_distance(data, reference, method):

    if method == 'R2':
        return linregress(data, reference)[2]

    elif method == 'SPEARMEN':
        return spearmanr(data, reference)[0]

    elif method == 'INTEGRALDIFF':
        distance = abs(np.array(data) - np.array(reference))
        return np.sum((np.max(distance) - distance)/(len(distance)*np.max(distance)))
    else:
        print('Invalid metric!')
        sys.exit()


def convolve(Sample, comp_mult=100):

    Convolved_sample = {col: [Sample[col][i] for i in  range(0, len(Sample[col]), comp_mult)] for col in Sample}

    return Convolved_sample



parser = argparse.ArgumentParser()
parser.add_argument('-sample', type=str, default=None, help='n-dim timeline')
parser.add_argument('-templates', type=str, default='templates', help='path to a folder containing templates')
parser.add_argument('-step', type=int, default=5000, help='step of search')
parser.add_argument('-threshold', type=float, default=0.7, help='matching threshold (from 0 to 1, default 0.7)')
parser.add_argument('-outdir', type=str, default='out', help='output directory')
parser.add_argument('-metric', type=str, default='R2', help='metric to calculate distance ("R2", "SPEARMEN", "INTEGRALDIFF"')
parser.add_argument('-conv_val', type=int, default=100, help='convolution parameter')
args = parser.parse_args()

if args.sample == None:
    print('choose sample!')
    sys.exit()

if args.metric not in methods:
    print('Invalid metric!')
    sys.exit()


Templates = []

els_templates = os.listdir(args.templates)

for template in els_templates:
    
    t = np.loadtxt(args.templates + '/' + template)
    t = np.array([[t[j][i] for i in range(0, len(t[j]), args.conv_val)] for j in range(len(t))])
    Templates.append(t)


    print(Templates[-1].shape)

Sample = pd.read_csv(args.sample, sep='\t', skiprows=3, header=None, names=[str(i) for i in range(17)], engine='python')


Sample1 = convolve(Sample, comp_mult=args.conv_val)

Template_matches = []
    
S_line = []
templates_matches = []


diff = int(20000/args.conv_val)

for i in range(0, len(Sample1['1']) - diff, int(args.step/args.conv_val)):

    templates_matches.append([])

    print('step {}'.format(int(i*args.conv_val/args.step)))
    print('{} to {} fragment:'.format(i*args.conv_val, (i + diff)*args.conv_val))
    for t in Templates:

        align_shifts = []
        for col in Sample1:
            if col == '16':
                continue
            
            S1 = list(Sample1[col][i:i + diff])
            S2 = t[int(col)]
        
            align = fastdtw(S1, S2, radius=10)

            el1_al = [S1[q[0]] for q in align[1]]
            el2_al = [S2[q[1]] for q in align[1]]
        
            align_shifts.append([q for q in align[1]])
            


        min_shift = min([len(q) for q in align_shifts])

        mean_shift1 = [np.median([j[q][0] for j in align_shifts]) for q in range(min_shift)]
        mean_shift2 = [np.median([j[q][1] for j in align_shifts]) for q in range(min_shift)]


        S1 = [list(Sample1[col])[i:i + diff] for col in Sample1 if col != '16']
        
        listSample1 = []
        listTemplate = []

        for col in range(len(S1)):
            for q in mean_shift1:
                listSample1.append(S1[col][int(q)])
            for q in mean_shift2:
                listTemplate.append(t[col][int(q)])



        S_line.append(spearmanr(listSample1, listTemplate)[0])


        
        print('\tSpearman: {}, {}'.format(*spearmanr(listSample1, listTemplate)))
        print('\tLinregress: {}, {}'.format(*linregress(listSample1, listTemplate)[2:4]))
        print()

        templates_matches[-1].append(compute_distance(listSample1, listTemplate, method=args.metric))
    


dists = []

plt.figure(figsize=(20, 10))
for t in range(len(templates_matches[0])):
    dists.append(np.array([i[t] for i in templates_matches if len(i) != 0]))

    plt.plot([d*args.step for d in range(len(dists[-1]))], dists[-1])



norm = int(20000/(args.step*4))


os.mkdir(args.outdir)

templates_coords = [[] for d in dists]
for dist in range(len(dists)):

    f_out = open(args.outdir + '/template_{}.txt'.format(dist+1), 'w')

    for point in range(len(dists[dist])):
        if dists[dist][point] <= args.threshold:
            continue

        skip = 0
        for p in range(max(0, point - norm), min(point + norm + 1, len(dists[dist]))):
            for d in range(len(dists)):
                if dists[d][p] > dists[dist][point]:
                    skip = 1 
                    break
        if skip == 0:
            templates_coords[dist].append((point*args.step, point*args.step+20000))
            f_out.write('{} '.format(dists[dist][point]))


colors = ['r', 'g', 'b']

current_c = 0
for t in templates_coords:
    for coord in t:
        plt.axvline(x=coord[0]-10000, c=colors[current_c])
        plt.axvline(x=coord[1]-10000, c=colors[current_c])
    
    current_c += 1



plt.savefig(args.outdir + '/templates_matching.png', dpi=300, format='png')
plt.show()
print(templates_coords)


current_template = 1
for t in templates_coords:
    for coord in t:

        Image = [[Sample[col][j] for j in range(coord[0], coord[1])] for col in Sample if col != '16']
        np.savetxt(args.outdir + '/template_{}_{}:{}.txt'.format(current_template, coord[0], coord[1]), Image)
        Image = [[Sample[col][j] for j in range(coord[0], coord[1], 200)] for col in Sample if col != '16']
        
        plt.imshow(Image)
        plt.savefig(args.outdir + '/template_{}_{}:{}.png'.format(current_template, coord[0], coord[1]), dpi=300, format='png')

    current_template += 1



    
