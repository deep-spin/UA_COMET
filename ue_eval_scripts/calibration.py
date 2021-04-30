from scipy.special import erfinv
import numpy as np
import itertools
from tqdm import tqdm

NUM_BINS = 100

def compute_calibration_error_non_parametric(target, scores, num_bins=NUM_BINS//5, scaling_val=1):
    matches = []
    gammas = np.linspace(0, 1, num_bins)
    scores = [np.array(sorted(i))/scaling_val for i in scores]
    for gamma in tqdm(gammas):   
        # scores = [np.array(sorted(i))/scaling_val for i in scores]
        lower = [np.quantile(s, (1-gamma)/2) for s in scores]
        upper = [np.quantile(s, (1+gamma)/2) for s in scores]
        correct = np.logical_and(lower <= target, target <= upper).sum()/len(target)  
        matches.append(correct)
        
    calibration_error = (np.abs(gammas - matches)).mean()
    return calibration_error, gammas, matches


def optimize_calibration_error_non_parametric(target, scores, scaling_vals, num_bins=NUM_BINS//5):
    best = np.inf
    best_scale = np.nan
    for scaling_val in tqdm(scaling_vals):
        calibration_error, _, _ = compute_calibration_error_non_parametric(
            target, scores, num_bins, scaling_val)
        if calibration_error < best:
            best_scale = scaling_val
            best = calibration_error
    return best, best_scale


def probit(p):
    return np.sqrt(2)*erfinv(2*p-1)


def compute_calibration_error(target, mean, std, std_sum=0, std_scale=1,
                              num_bins=NUM_BINS):
    matches = []
    gammas = np.linspace(0, 1, num_bins)
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    for gamma in gammas:
        lower = mean + std_transformed * probit((1-gamma)/2)
        upper = mean + std_transformed * probit((1+gamma)/2)
        correct = np.logical_and(
            lower <= target, target <= upper).sum() / len(target)
        matches.append(correct)

    calibration_error = (np.abs(gammas - matches)).mean()
    return calibration_error, gammas, matches


def optimize_calibration_error(target, mean, std, std_sums, std_scales,
                               num_bins=NUM_BINS):
    best = np.inf
    best_std_sum = np.nan
    best_std_scale = np.nan
    for (std_sum, std_scale) in tqdm(itertools.product(std_sums, std_scales)):
    #for std_sum, std_scale in zip(std_sums, std_scales):
        calibration_error, _, _ = compute_calibration_error(
            target, mean, std, std_sum, std_scale, num_bins)
        if calibration_error < best:
            best_std_sum = std_sum
            best_std_scale = std_scale
            best = calibration_error
    return best, best_std_sum, best_std_scale
