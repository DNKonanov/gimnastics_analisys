import numpy as np


def convolve(deconvSample, conv_val=100):


    Convolved_sample = [[col[i] for i in range(0, len(col), conv_val)] for col in deconvSample.T]

    return np.array(Convolved_sample)



def deconvolve(convSample, conv_val=100):
    
    DeconvolvedSample = []

    for i in convSample.T:
        for q in range(conv_val):
            DeconvolvedSample.append(i)
    
    DeconvolvedSample = np.array(DeconvolvedSample).T
    return DeconvolvedSample



