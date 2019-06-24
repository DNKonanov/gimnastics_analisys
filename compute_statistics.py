from scipy import std
import argparse
import sys
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('-input', default=None, type=str, help='input file')

args = parser.parse_args()

if args.input == None:
    print('select input file!')
    sys.exit()

vals = np.loadtxt(args.input)

print('SD: {}'.format(np.std(vals)))
print('Mean: {}'.format(np.mean(vals)))



