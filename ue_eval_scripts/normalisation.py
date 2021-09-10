import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from calibration import compute_calibration_error, optimize_calibration_error
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import pearsonr

from os import listdir
from os.path import isfile, join


def compute_z_norm(scores):
    mean=0
    std=1
    np_scores = np.array([val for _,sys in scores.items() for _,doc in sys.items() for val in doc])
    print(np_scores.shape)
    scores_tbn = np_scores.flatten()
    std= np.std(scores_tbn)
    mean = np.mean(scores_tbn)
    return mean, std

def compute_z_norm_sent(scores):
    mean=0
    std=1

    #np_scores = np.array([val for d in pre_scores for (key,val) in d.items() ])
    #np_scores = np.array([val for _,sys in scores.items() for _,doc in sys.items() for val in doc])
    np_scores = np.array([val for _,sys in scores.items() for _,doc in sys.items() for _,val in doc.items() ])
    
    print(np_scores.shape)
    
    # if we have multi-dimensional array inscores (e.g. output of MCD) flatten first
    scores_tbn = np_scores.flatten()
    std= np.std(scores_tbn)
    mean = np.mean(scores_tbn)
    return mean, std

def compute_fixed_std(comet_scores_mean, da_scores):
    n = len(comet_scores_mean)
    assert(len(comet_scores_mean)==len(da_scores))
    sigma_sq = (da_scores - comet_scores_mean)**2
    sigma = np.sqrt(np.sum(sigma_sq)/n)
    return sigma

def compute_fixed_ci(comet_scores_mean, da_scores):
    n = len(comet_scores_mean)
    assert(len(comet_scores_mean)==len(da_scores))
    sigma_sq = (da_scores - comet_scores_mean)**2
    sigma = np.sqrt(np.sum(sigma_sq)/n)
    return sigma

def standardize(scores_test, scores_dev, norm):
    norm_mean = 0.0
    norm_std = 1.0
    if norm:
        norm_mean, norm_std = compute_z_norm(scores_dev)
    all_scores = np.array([val for _,sys in scores_test.items() for _,doc in sys.items() for val in doc ])
    all_scores -= norm_mean
    all_scores /= norm_std
    return all_scores