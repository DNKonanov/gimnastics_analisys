from scipy.stats import mannwhitneyu
import numpy as np
import argparse
import sys
parser = argparse.ArgumentParser()

parser.add_argument('-input1', type=str, default=None, help='input 1')
parser.add_argument('-input2', type=str, default=None, help='input 2')
parser.add_argument('-pval', type=float, default=0.05, help='p-value (default 0.05)')

args = parser.parse_args()

if None in [args.input1, args.input2]:
    print('select inputs!')
    sys.exit()

X = np.loadtxt(args.input1)
Y = np.loadtxt(args.input2)

mw = mannwhitneyu(X, Y)

p_value = mw[1]
print(p_value)
if p_value > args.pval:
    print('reject!')
else:
    print('good!')


