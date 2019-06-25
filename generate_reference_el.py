import numpy as np
import argparse
import sys
import os
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-inputdir', type=str, default=None, help='inputs directory')
parser.add_argument('-outdir', type=str, default=None, help='outs directory (default the same as input directory)')
parser.add_argument('-reference_input', type=str, default=None, help='reference_input (default is first file from input directory')
parser.add_argument('-radius', type=int, default=1, help='size of neighborhood when expanding the path in the FastDTW method')


args = parser.parse_args()


def convolve(Sample, comp_mult=100):

    Convolved_sample = [[i[q] for q in range(0, len(i), comp_mult)] for i in Sample]

    return np.array(Convolved_sample)


def create_correction_dist(inputs):
    
    Correction_dist = np.array([])
    return Correction_dist


def align_inputs(inputs, deconv_reference_input, radius):

    aligned_inputs = []

    reference_input = convolve(deconv_reference_input)

    for deconvInput in inputs:
        Input = convolve(deconvInput)

        path = fastdtw(Input.T, reference_input.T, radius=args.radius)[1]

        print(path)
        step = 0
        aligned_input = []

        al_ref = []
        al_input = []
        for p in path:
            al_ref.append(reference_input.T[p[1]])
            al_input.append(Input.T[p[0]])

        al_ref = np.array(al_ref).T
        al_input = np.array(al_input).T


        fig, axs = plt.subplots(4,1)


        axs[0].imshow(al_ref)
        axs[1].imshow(al_input)
        axs[2].imshow(Input)
        axs[3].imshow(reference_input)
        plt.show()
        

    aligned_inputs = np.array(aligned_inputs)

    return aligned_inputs


def generate_avg_input(align_inputs):

    
    _transposed_inputs = align_inputs.T

    Avg_input = []

    for raw in _transposed_inputs:
        Avg_input.append(np.mean(raw))
    
    Avg_input = np.array([])

    
    return Avg_input


def generate_reference(inputs, reference_input, radius):


    aligned_inputs = align_inputs(inputs, reference_input, radius)

    Correction_dist = create_correction_dist(aligned_inputs)
    Avg_input = generate_avg_input(aligned_inputs)

    return Correction_dist, Avg_input

if args.inputdir == None:
    print('select input directory!')
    sys.exit()

if args.outdir == None:
    outdir = args.inputdir
else:
    outdir = args.outdir
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

try:
    Files = [np.loadtxt(args.inputdir + '/' + f) for f in os.listdir(args.inputdir)]

except:
    print('check that input directory is exist!')
    sys.exit()

if args.reference_input != None:
    reference_input = np.loadtxt(args.inputdir + '/' + args.reference_input)
else:
    try:
        reference_input = Files[0]
    except:
        print('Invalud input!')
        sys.exit()


print('Start computing...')
Correction_dist, Avg_input = generate_reference(Files, reference_input, args.radius)



np.savetxt(outdir + '/correction_dist.txt', Correction_dist)
np.savetxt(outdir + '/avg_input.txt', Avg_input)
print('Complete!')