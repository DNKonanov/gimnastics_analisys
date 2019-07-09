from gymnlib.alignment import align_samples, align_multisamples
from gymnlib.convolution import convolve, deconvolve
from gymnlib.extracting import parse_templates, generate_reference
import numpy as np
import argparse
import sys
import os
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('-inputdir', type=str, default=None, help='inputs directory')
parser.add_argument('-outdir', type=str, default='av_templates', help='outs directory (default "av_tempaltes)')
parser.add_argument('-outname', type=str, default='el1', help='name of the template (default is el1)')
parser.add_argument('-reference_input', type=str, default=None, help='reference_input (default is first file from input directory')
parser.add_argument('-radius', type=int, default=1, help='size of neighborhood when expanding the path in the FastDTW method')
parser.add_argument('-conv_val', type=int, default=100, help='convolution value')
parser.add_argument('-create_correction', type=int, default=0, help='create correction matrix flag (0 or 1)')

args = parser.parse_args()


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
Correction_dist, Avg_input = generate_reference(Files, reference_input, args.radius, conv=args.conv_val)

fig, axs = plt.subplots(2,1)

axs[0].imshow(Avg_input)
axs[0].set_title('Average')
axs[1].imshow(Correction_dist)
axs[1].set_title('Variance')
plt.show()


if args.create_correction == 0:
    Correction_dist = np.zeros(Correction_dist.shape) + 1




try:
    os.mkdir(outdir + '/templates')
    os.mkdir(outdir + '/corrections')
except FileExistsError:
    pass

np.savetxt(outdir + '/templates/{}.txt'.format(args.outname), deconvolve(Avg_input, conv_val=args.conv_val))
np.savetxt(outdir + '/corrections/{}.txt'.format(args.outname), deconvolve(Correction_dist, conv_val=args.conv_val))
print('Complete!')