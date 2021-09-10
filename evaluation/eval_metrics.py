import numpy as np
from scipy.stats import norm

def compute_avgll(target, mean, std, std_sum=0, std_scale=1):
    eps = np.finfo(float).eps
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    probs = [norm.pdf(q, mean[i], std_transformed[i]) for i, q in enumerate(target)]
    avgll = -np.log(probs+eps).mean()
    negll = -np.log(probs+eps).sum()
    return avgll, negll