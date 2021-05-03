import numpy as np
from scipy.stats import norm

def compute_avgll(target, mean, std):
    eps = np.finfo(float).eps
    probs = [norm.pdf(q, mean[i], std[i]) for i, q in enumerate(target)]
    avgll = -np.log(probs+eps).mean()
    negll = -np.log(probs+eps).sum()
    return avgll, negll