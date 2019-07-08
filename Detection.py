import librosa
import sys
import numpy as np
from numpy.linalg import norm

templ = sys.argv[1]
file = sys.argv[2]
y_template, sr_template = librosa.load(templ)
y, sr = librosa.load(file)
y_template = (y_template - y_template.mean()) / y_template.std()
y = (y - y.mean()) / y.std()
mfcc_template = librosa.feature.mfcc(y=y_template, sr=sr_template,
                                     dct_type=1, n_mfcc=50, norm=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr,
                            dct_type=1, n_mfcc=50, norm=None)


def compare(X, Y):
    scalar = np.sum(X*Y, axis=0)
    cos = scalar / (norm(X, axis=0) * norm(Y, axis=0))
    return np.mean(cos)


def search(MFCC1, MFCC2):
    L1 = MFCC1.shape[1]
    L2 = MFCC2.shape[1]
    MAX = -999
    if L1 > L2:
        Big = MFCC1
        Small = MFCC2
        for i in range(L1-L2+1):
            MAX = max(MAX, compare(Small, Big[:, i:i+L2]))
    elif L2 > L1:
        Big = MFCC2
        Small = MFCC1
        for i in range(L2-L1+1):
            MAX = max(MAX, compare(Small, Big[:, i:i+L1]))
    else:
        return compare(MFCC1, MFCC2)
    return MAX


score = search(mfcc_template, mfcc)
if score > 0.7:
    print('The file contains this template.')
else:
    print('The file does not contain this template.')
