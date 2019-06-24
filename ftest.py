from scipy.stats import f
import numpy as np
import argparse
import sys
parser = argparse.ArgumentParser()

parser.add_argument('-input1', type=str, default=None, help='input 1')
parser.add_argument('-input2', type=str, default=None, help='input 2')
parser.add_argument('-pval', type=float, default=0.05, help='p-value')

args = parser.parse_args()

if None in [args.input1, args.input2]:
    print('select inputs!')
    sys.exit()

X = np.loadtxt(args.input1)
Y = np.loadtxt(args.input2)

F = np.var(X)/np.var(Y)

alpha = args.pval
p_value = 1 - f.sf(F, len(X)-1, len(Y)-1)
print(p_value)
if p_value > alpha:
    print('reject!')
else:
    print('got it!')
    # Reject the null hypothesis that Var(X) == Var(Y)


