from scipy.special import erfinv
import numpy as np
import itertools
from tqdm import tqdm
import math

NUM_BINS = 100

def compute_calibration_error_non_parametric(target, scores, num_bins=NUM_BINS//5, scaling_val=1, scaling_sum=0):
    matches = []
    gammas = np.linspace(0, 1, num_bins)
    scores = [(np.array(sorted(i))/scaling_val)+scaling_sum for i in scores]
    for gamma in tqdm(gammas):   
        # scores = [np.array(sorted(i))/scaling_val for i in scores]
        lower = [np.quantile(s, (1-gamma)/2) for s in scores]
        upper = [np.quantile(s, (1+gamma)/2) for s in scores]
        correct = np.logical_and(lower <= target, target <= upper).sum()/len(target)  
        matches.append(correct)
        
    calibration_error = (np.abs(gammas - matches)).mean()
    return calibration_error, gammas, matches


def optimize_calibration_error_non_parametric(target, scores, scaling_vals, scaling_sums, num_bins=NUM_BINS//5):
    best = np.inf
    best_scale = np.nan
    best_sum = np.nan
    for (scaling_sum, scaling_val) in tqdm(itertools.product(scaling_sums, scaling_vals)):
        calibration_error, _, _ = compute_calibration_error_non_parametric(
            target, scores, num_bins, scaling_val, scaling_sum)
        if calibration_error < best:
            best_scale = scaling_val
            best_sum = scaling_sum
            best = calibration_error
    return best, best_scale, best_sum


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


# from https://arxiv.org/pdf/2005.12496.pdf 
def compute_sharpness(std, std_sum=0, std_scale=1):
    
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    sharpness = np.mean(std_transformed**2)
    return sharpness


# from https://openreview.net/pdf?id=ryg8wpEtvB
def compute_ence(target, mean, std, std_sum=0, std_scale=1,
                              num_bins=100):
    matches = []
    gammas = np.linspace(0, len(target) , num_bins+1)
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    sorted_idxs = np.argsort(std_transformed)
    std_sorted = [std_transformed[i] for i in sorted_idxs]
    mean_sorted = [mean[i] for i in sorted_idxs]
    target_sorted = [target[i] for i in sorted_idxs]
    for i,_ in enumerate(gammas):
        if i+1<len(gammas):
            lower = math.floor(gammas[i])
            upper = math.floor(gammas[i+1])
            bin_mean = np.asarray(mean_sorted[lower:upper])
            bin_target = np.asarray(target_sorted[lower:upper])
            bin_std = np.asarray(std_sorted[lower:upper])
            
            width = upper-lower
            epsilon=0.001
            if not width>0.0:
                width = epsilon
            
            rmse = np.sqrt(1/width * np.sum((bin_mean-bin_target)**2))
            mvar = np.sqrt(1/width * np.sum(bin_std**2))
            nse = np.abs((mvar-rmse)/mvar)
            matches.append(nse)

    ence = np.mean(matches)
    
    return ence, np.linspace(1, 100 , num_bins), matches


# from https://openreview.net/pdf?id=ryg8wpEtvB
def compute_ence_rn(target, mean, std, std_sum=0, std_scale=1,
                              num_bins=100):
    matches = []
    gammas = np.linspace(0, len(target) , num_bins+1)
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    sorted_idxs = np.argsort(std_transformed)
    std_sorted = [std_transformed[i] for i in sorted_idxs]
    mean_sorted = [mean[i] for i in sorted_idxs]
    target_sorted = [target[i] for i in sorted_idxs]
    for i,_ in enumerate(gammas):
        if i+1<len(gammas):
            lower = math.floor(gammas[i])
            upper = math.floor(gammas[i+1])
            bin_mean = np.asarray(mean_sorted[lower:upper])
            bin_target = np.asarray(target_sorted[lower:upper])
            bin_std = np.asarray(std_sorted[lower:upper])
            
            width = upper-lower
            epsilon=0.001
            if not width>0.0:
                width = epsilon
            rmse = np.sqrt(1/width * np.sum((bin_mean-bin_target)**2))
            mvar = np.sqrt(1/width * np.sum(bin_std**2))
            nse = np.abs((mvar-rmse)/rmse)
            matches.append(nse)

    ence_rn = np.mean(matches)
    
    return ence_rn, np.linspace(1, 100 , num_bins), matches

# from https://openreview.net/pdf?id=ryg8wpEtvB
def compute_ence_nn(target, mean, std, std_sum=0, std_scale=1,
                              num_bins=100):
    matches = []
    gammas = np.linspace(0, len(target) , num_bins+1)
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    sorted_idxs = np.argsort(std_transformed)
    std_sorted = [std_transformed[i] for i in sorted_idxs]
    mean_sorted = [mean[i] for i in sorted_idxs]
    target_sorted = [target[i] for i in sorted_idxs]
    for i,_ in enumerate(gammas):
        if i+1<len(gammas):
            lower = math.floor(gammas[i])
            upper = math.floor(gammas[i+1])
            bin_mean = np.asarray(mean_sorted[lower:upper])
            bin_target = np.asarray(target_sorted[lower:upper])
            bin_std = np.asarray(std_sorted[lower:upper])
            
            width = upper-lower
            epsilon=0.001
            if not width>0.0:
                width = epsilon
            
            rmse = np.sqrt(1/width * np.sum((bin_mean-bin_target)**2))
            mvar = np.sqrt(1/width * np.sum(bin_std**2))
            nse = np.abs((mvar-rmse))
            matches.append(nse)

    ence_nn = np.mean(matches)
    
    return ence_nn, np.linspace(1, 100 , num_bins), matches

# From https://arxiv.org/pdf/2006.10255.pdf
def compute_ecpe(target, mean, std, std_sum=0, std_scale=1,
                              num_bins=100):
    calibration_error, gammas, matches = compute_calibration_error(
        target, mean, std, std_sum, std_scale, num_bins)
    return calibration_error, gammas, matches

def compute_mcpe(target, mean, std, std_sum=0, std_scale=1,
                              num_bins=100):
    matches = []
    gammas = np.linspace(0, 1, num_bins)
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    for gamma in gammas:
        lower = mean + std_transformed * probit((1-gamma)/2)
        upper = mean + std_transformed * probit((1+gamma)/2)
        correct = np.logical_and(
            lower <= target, target <= upper).sum() / len(target)
        matches.append(correct)

    calibration_error = np.max((np.abs(gammas - matches)))
    return calibration_error, gammas, matches

# sharpness related
def compute_epiw(mean, std, std_sum=0, std_scale=1,
                              num_bins=100):

    matches = []
    gammas = np.linspace(0, 1, num_bins)
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    for gamma in gammas:
        lower = mean + std_transformed * probit((1-gamma)/2)
        upper = mean + std_transformed * probit((1+gamma)/2)
        width = upper-lower
        msk = np.ma.masked_invalid(width)
        width = np.ma.filled(msk, fill_value = 10)
        matches.append(width.mean())
   
    sharpness = np.ma.masked_invalid(matches).mean()
    
    return sharpness, gammas, matches

def compute_mpiw(mean, std, std_sum=0, std_scale=1,
                              num_bins=100):
    matches = []
    gammas = np.linspace(0, 1, num_bins)
    std_transformed = np.sqrt(std_sum**2 + (std_scale*std)**2)
    for gamma in gammas:
        lower = mean + std_transformed * probit((1-gamma)/2)
        upper = mean + std_transformed * probit((1+gamma)/2)
        width = upper-lower
        msk = np.ma.masked_invalid(width)
        width = np.ma.filled(msk, fill_value = 10)
        matches.append(width.max())

    sharpness = np.ma.masked_invalid(matches).max()
    
    return sharpness, gammas, matches


