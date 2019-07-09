from scipy.stats import spearmanr, linregress
import numpy as np
import sys
import matplotlib.pyplot as plt

METRICS = ['LINREGRESS', 'SPEARMAN', 'INTEGRALDIFF']


def compute_distance(sample1, sample2, metric='LINREGRESS', correction=None, distance=1):



    if metric == METRICS[0]:
        return linregress((sample1/(correction+1)).flatten(), (sample2/np.sqrt(correction+1)).flatten())[2]/distance

    elif metric == METRICS[1]:
        return spearmanr((sample1/(correction+1)).flatten(), (sample2/np.sqrt(correction+1)).flatten())[0]/distance

    elif metric == METRICS[3]:
        distance = abs(np.array(sample1) - np.array(sample2))
        return np.sum((np.max(distance) - distance)/(len(distance)*np.max(distance)))
    else:
        print('Invalid metric!')
        return None