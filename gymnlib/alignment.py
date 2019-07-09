import sys
from fastdtw import fastdtw
from .convolution import convolve
import numpy as np
ALIGNMENT_TYPES = ['SHIFT', 'REPLACE']

def align_samples(sample0, sample1, radius=10, al_type='SHIFT'):

    distance, path = fastdtw(sample0.T, sample1.T, radius=radius)

    if al_type == ALIGNMENT_TYPES[0]:

        aligned_sample0 = []
        aligned_sample1 = []

        for p in path:
            aligned_sample0.append(sample0.T[p[0]])
            aligned_sample1.append(sample1.T[p[1]])

    if al_type == ALIGNMENT_TYPES[1]:

        aligned_sample0 = sample0.T
        aligned_sample1 = [] 

        step = 0
        for p in path:
            if p[0] == step:
                aligned_sample1.append(sample1.T[p[1]])
                step += 1
        aligned_sample1 = np.array(aligned_sample1)

    return np.array(aligned_sample0), np.array(aligned_sample1), distance


def align_multisamples(inputs, deconv_reference_input, radius=10, conv=100):
    aligned_inputs = []

    reference_input = convolve(deconv_reference_input.T)

    for deconvInput in inputs:
        Input = convolve(deconvInput.T)
        
        path = fastdtw(Input.T, reference_input.T, radius=radius)[1]


        
        al_input = []

        step = 0
        for p in path:
            
            if p[1] == step:
                al_input.append(Input.T[p[0]])
                step += 1

        al_input = np.array(al_input).T

        aligned_inputs.append(al_input)
        

    

    return aligned_inputs