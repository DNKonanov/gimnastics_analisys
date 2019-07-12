import numpy as np
import os
from .convolution import convolve
from fastdtw import fastdtw
from .alignment import align_multisamples

METRICS = ['LINREGRESS', 'SPEARMAN', 'INTEGRALDIFF']


def create_correction_dist(inputs):
    
    
    Correction_dist = np.zeros(inputs[0].shape)

    for i in range(len(inputs[0])):
        for j in range(len(inputs[0][i])):
            Correction_dist[i][j] = np.var([el[i][j] for el in inputs])

    return Correction_dist


def generate_avg_input(aligned_inputs):

    
    

    Avg_input = np.zeros(aligned_inputs[0].shape)

    for t in aligned_inputs:
        Avg_input += t
    
    Avg_input = Avg_input/len(aligned_inputs)

    return Avg_input


def generate_reference(inputs, reference_input, radius, conv=100):


    aligned_inputs = align_multisamples(inputs, reference_input, radius=radius, conv=conv)

    print([i.shape for i in aligned_inputs])

    Correction_dist = create_correction_dist(aligned_inputs)
    Avg_input = generate_avg_input(aligned_inputs)


    return Correction_dist, Avg_input



def parse_templates(templates_folder, templateslist, conv_val=100):

    Templates = []

    if templateslist != None:
        templateslist = templateslist.split(':')


    print(templateslist)
    Templates_corrections = []
    els_templates = os.listdir(templates_folder + '/templates')

    
    for template in els_templates:
        
        if templateslist != None and template not in templateslist:
            continue

        t = np.loadtxt(templates_folder + '/templates/' + template)
        t = np.array([[t[j][i] for i in range(0, len(t[j]), conv_val)] for j in range(len(t))])
        Templates.append(t)


        Templates_corrections.append(convolve(np.loadtxt(templates_folder + '/corrections/' + template).T, conv_val=conv_val).T)

        print(Templates[-1].shape)


    Templates_sizes = {i: len(Templates[i][0])*conv_val for i in range(len(Templates))}
    Templates = {i: Templates[i] for i in range(len(Templates))}

    return Templates, Templates_sizes, Templates_corrections



