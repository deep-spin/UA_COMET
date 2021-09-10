import json
import csv
import numpy as np
from scipy import stats
from calibration import *
import pandas as pd
from eval_metrics import *
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import pearsonr
import argparse
import itertools
from os import listdir
from os.path import isfile, join
from normalisation import *
from sklearn.metrics import ndcg_score, dcg_score

def optimise_prec(q_range, norm_human_avg_dev, norm_comet_avg_dev, comet_std_dev, sample_index_dev, bin_target, mean_hd, std_hd, relevant_p):    
    precision_by_bins = []
    precision = [5, 10, 15, 20, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    #srt_truth = sorted(range(len(norm_human_avg_dev_sort)), key=lambda k: norm_human_avg_dev_sort[k])
    
    for q in tqdm(q_range):
        threshold = q#(q-mean_hd)/std_hd
        comet_probs = calculate_score_cdfmax(norm_comet_avg_dev, comet_std_dev, threshold)
        #rank by index
        srt_comet_probs, sorted_comet_truth = (list(t) for t in zip(*sorted(zip(comet_probs, sample_index_dev), reverse=True)))
        #srt_comet_probs = sorted(range(len(comet_probs)), key=lambda k: comet_probs[k])
        
        #if (sum(comet_probs)>0.0 and len([i for i in comet_probs if i <= 0.0])<(len(comet_probs)//100)):
        #print(comet_probs[0:30])
        precision_vals=[]
        for prec in precision:
            n=prec
            precision_vals.append(compute_precision(sorted_comet_truth, bin_target, n))
        precision_avg = np.mean(precision_vals)
        precision_by_bins.append(precision_vals)
        #else:
        #    recall_by_bins.append(0.0)
    max_per_recall=np.argmax(precision_by_bins,axis=0)
    print(max_per_recall)
    max_per_recall= max_per_recall.tolist()
    max_per_recall=[i for i in max_per_recall if i != 0]
    max_idx = max(max_per_recall,key=max_per_recall.count)
    
    #max_idx = recall_by_bins.index(max(recall_by_bins))
    
    return q_range[max_idx]


#def optimise_q(q_range,norm_human_avg_dev_sort,norm_comet_avg_dev_sort, comet_std_dev_sort, mean_hd, std_hd, relevant_p):
def optimise_q(q_range, norm_human_avg_dev, norm_comet_avg_dev, comet_std_dev, sample_index_dev, bin_target, mean_hd, std_hd, relevant_p):    
    recall_by_bins = []
    recall = [10, 15, 20, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    #srt_truth = sorted(range(len(norm_human_avg_dev_sort)), key=lambda k: norm_human_avg_dev_sort[k])
    bin_target=bin_target[:relevant_p]
    for q in tqdm(q_range):
        #print(q)
        #threshold = (q-mean_hd)/std_hd
        threshold=q
        comet_probs = calculate_score_cdf(norm_comet_avg_dev, comet_std_dev, threshold)
        #rank by index
        srt_comet_probs, sorted_comet_truth = (list(t) for t in zip(*sorted(zip(comet_probs, sample_index_dev), reverse=True)))
        #srt_comet_probs = sorted(range(len(comet_probs)), key=lambda k: comet_probs[k])
        
        #if (sum(comet_probs)>0.0 and len([i for i in comet_probs if i <= 0.0])<(len(comet_probs)//100)):
        #print(srt_comet_probs[0:30])
        recall_vals=[]
        for rec in recall:
            n=rec
            recall_vals.append(compute_recall(sorted_comet_truth, bin_target, n))
        recall_avg = np.mean(recall_vals)
        recall_by_bins.append(recall_vals)
        #else:
        #    recall_by_bins.append(0.0)
    #print(recall_by_bins)
    
    max_per_recall=np.argmax(recall_by_bins,axis=0)
    print(max_per_recall)
    max_per_recall= max_per_recall.tolist()
    max_per_recall=[i for i in max_per_recall if i != 0]
    max_idx = max(max_per_recall,key=max_per_recall.count)
    
    #max_idx = recall_by_bins.index(max(recall_by_bins))
    
    return q_range[max_idx]

def calculate_score_cdfmax(means, stds, bin_thres):
    probs = [norm.sf(bin_thres, means[i], stds[i]) for i, _ in enumerate(means)]
    return probs

def calculate_score_cdf(means, stds, bin_thres):
    probs = [norm.cdf(bin_thres, means[i], stds[i]) for i, _ in enumerate(means)]
    return probs

def calculate_score_sub(means, stds):
    score = [means[i]*(1- stds[i]) for i, _ in enumerate(means)]
    return score

def compute_precision(predictions, target, n):
    #receives lists of indexes
    n_preds = predictions[:n]
    relevant = 0
    #print(len(target))
    for pred in n_preds:
        if pred in target: 
            relevant+=1
    return relevant/n


def compute_recall(predictions, target, n):
    #receives lists of indexes
    predictions=predictions[:n]
    n_preds = target
    relevant = 0
    for tgt in n_preds:
        if tgt in predictions:
            relevant+=1
    #print(relevant)
    return relevant/len(target)

def compute_ndcg(predictions, target, n):
    #receives lists of indexes
    assert(len(predictions)==len(target))
    l = len(target)
 
    ndcg = ndcg_score([target], [predictions],  k=n, ignore_ties=True)
    return ndcg

def compute_dcg(predictions, target, n):
    #receives lists of indexes
    assert(len(predictions)==len(target))
    l = len(target)
    #predictions = [(l- x)/100 for x in predictions]
    #target = [(l- x)/100 for x in target]
    #print(target)
    dcg = dcg_score([target], [predictions],  k=n, ignore_ties=True)
    return dcg

def compute_ap(predictions, target, n):
    ap = 0
    sum = 0
    for i, pred in enumerate(predictions[:n]):
        if pred in target[:n]:
            sum+=1
            ap+=sum/(i+1)
        else:
            ap+=sum/(i+1)
    return ap/n






