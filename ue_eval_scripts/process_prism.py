import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pandas as pd

from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import pearsonr
import argparse
import itertools
from os import listdir
from os.path import isfile, join


def find_idx(sent, sents):
    index = -1
    tmp = sent[:15]
    for i, s in enumerate(sents):
        if s.startswith(tmp):
            return i
    print('not found')
    print(sent)

    exit(0)
    return index

def reorder(s_sents, h_sents, ordered_file):
    ordered = []
    reordered_src=[]
    reordered_tgt=[]
    with open(ordered_file, 'r') as ord:
        for line in ord:
            ordered.append(line.strip())
            tmp = line.strip().encode('utf8').decode('utf8')
            try:
                i = s_sents.index(tmp)
            except:
                print(tmp)
                i = find_idx(tmp, s_sents)
            reordered_src.append(s_sents[i])
            reordered_tgt.append(h_sents[i])
    return reordered_src, reordered_tgt




def detokenize(sentence):
    #print(sentence)
    subwords = sentence.split(' ')
    sentence_str = ''
    for subword in subwords:
        if subword.strip().startswith('▁'):
            #print(subword)
            subword_i = subword.replace('▁',' ')
            #print(subword_i)
            sentence_str = sentence_str+ subword_i
        else:
            sentence_str = sentence_str+subword
    sentence_str = sentence_str.strip()
    return sentence_str


def parse_prism_out(file):
    s_sents=[]
    t_sents=[]
    with open(file, 'r') as prism_out:
        
        for line in prism_out:
            if line.startswith('S-'):
                sentence = line.split('\t')[1]
                proper_sent = detokenize(sentence)
                s_sents.append(proper_sent)
            elif line.startswith('H-'):
                sentence = line.split('\t')[2][5:]
                proper_sent = detokenize(sentence)
                t_sents.append(proper_sent)
    
    return s_sents, t_sents


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process comet outputs')
    parser.add_argument('--prism-file', type=str, 
                        help='path to comet setup to test on')
    parser.add_argument('--output-file', type=str, 
                        help='path to scores for testing on')
    parser.add_argument('--ordered-file', type=str, 
                        help='path to comet setup to test on')
    parser.add_argument('--alpha', type=str, 
                        help='alpha')
  

    args = parser.parse_args()
    
    s_sents, h_sents = parse_prism_out(args.prism_file)
    s_sents, h_sents = reorder(s_sents, h_sents, args.ordered_file)
    s_file = open(args.output_file+'prism_'+args.alpha+'_src.txt', 'w')
    h_file = open(args.output_file+'prism_'+args.alpha+'_par.txt', 'w')
    for sent in s_sents:
        s_file.write(sent+'\n')
    for sent in h_sents:
        h_file.write(sent+'\n')
    s_file.close()
    h_file.close()

