import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
from ir_functions import *
from sklearn.metrics import average_precision_score

def get_df(comet_dir, da_dir, nruns=100, docs=False, ens=True):
  
    SETUP_PATH = comet_dir
    files = [f for f in listdir(SETUP_PATH) if isfile(join(SETUP_PATH, f))]
    sys_files = [f for f in files if (f.split('_')[0] == 'system') and ('Human' not in f)]
    da_scores = pd.read_csv(da_dir)
    da_scores.system = da_scores.system.apply(lambda x: x.split('.')[0])

    dfs = []

    for s in sys_files:
        f = open(join(SETUP_PATH,s), 'r')
        data = json.loads(f.read()) 
        f.close() 
        system_name = '_'.join(s.split('.')[0].split('_')[1:])
        lines = [[i['src'], i['mt'], i['ref'], i['dp_runs_scores']] for i in data if 'dp_runs_scores' in i.keys()]
        df_ = pd.DataFrame(data=np.array(lines), columns=['src','mt', 'ref', 'dp_runs_scores'])
        da_scores_ = da_scores[da_scores.system == system_name]
        df = df_.merge(da_scores_, how='inner', on=['src', 'mt'])
        
        df['dp_runs_scores'] = df['dp_runs_scores'].apply(lambda x: x[:nruns])
        df['predicted_score_mean'] = df['dp_runs_scores'].apply(lambda x: np.mean(x)) # segment-level
        df['predicted_score_std'] = df['dp_runs_scores'].apply(lambda x: np.std(x)) # segment-level
        df['q-mu'] = np.abs(df['human_score'] - df['predicted_score_mean'])
        dfs.append(df)
    df_all = pd.concat(dfs)
    df_all.reset_index(inplace=True)
    return df_all


def get_dfs(comet_dir, score_file, len_calc, nruns=100,  type='mqm', lp='en-de'):
    mqms = pd.read_csv(score_file) 
    mqms = mqms[mqms['lp']== lp]
    SETUP_PATH = comet_dir
    files = [f for f in listdir(SETUP_PATH) if isfile(join(SETUP_PATH, f))]
    sys_files = [f for f in files if f.split('_')[0] == 'system']
    print(sys_files)
    dfs=[]
    mqms.system = mqms.system.apply(lambda x: x.split('.')[0])
   
    for s in sys_files:
        f = open(join(SETUP_PATH,s), 'r')
        data = json.loads(f.read()) 
        f.close() 
        system_name = '_'.join(s.split('.')[0].split('_')[1:])
        if system_name!="Human-A":
            lines = [[i['src'], i['mt'], i['ref'], i['dp_runs_scores']] for i in data if 'dp_runs_scores' in i.keys()]
            df_ = pd.DataFrame(data=np.array(lines), columns=['src','mt', 'ref', 'dp_runs_scores'])
            mqms_ = mqms[mqms.system == system_name]
            print('..........')
            print(system_name)
            print(len(mqms_))
            print(len(df_))
            df = df_.merge(mqms_, how='inner', on=['src', 'mt'])
            print(len(df))
            df['dp_runs_scores'] = df['dp_runs_scores'].apply(lambda x: x[:nruns])
            df['predicted_score_mean'] = df['dp_runs_scores'].apply(lambda x: np.mean(x)) # segment-level
            df['predicted_score_std'] = df['dp_runs_scores'].apply(lambda x: np.std(x)) # segment-level
            df['q-mu'] = np.abs(df['score'] - df['predicted_score_mean'])
            df['lengths'] = df['mt'].str.len()
            #convert scores to an ascending scale to be aligned with COMET ranking (the higher the better)
            if len_calc:
                df['human_score'] = np.abs((100 - df['score']/df['lengths'])) #normalising by length 
            else:
                df['human_score'] = np.abs(100 - df['score'])  
           
            dfs.append(df)
        
    df_all = pd.concat(dfs)
    df_all.reset_index(inplace=True)
    
    if type=='mqm':
        df_all.rename(columns={'seg_id':'segment_id'}, inplace=True) 
        #df_all.rename(columns={'score':'human_score'}, inplace=True)
    return df_all



def get_system_scores_human(df):
    systems = {}
    for _, row in df.iterrows():
        print(row)
        system = row['system']
        if not system in systems:
            systems[system]={}
        doc_id = row['doc_id']
        if not doc_id in systems[system]:
            systems[system][doc_id]={}
        segment_id = row['segment_id']
        if not segment_id in systems[system][doc_id]:
            systems[system][doc_id][segment_id]=[]
        score =  row['human_score']
        systems[system][doc_id][segment_id].append(score)
    return systems


def get_system_scores_comet(df):
   
    systems = {}
    for idx, row in df.iterrows():
        print(row)
        system = row['system']
        if not system in systems:
            systems[system]={}
        doc_id = row['doc_id']
        if not doc_id in systems[system]:
            systems[system][doc_id]={}
        segment_id = row['segment_id']
        if not segment_id in systems[system][doc_id]:
            systems[system][doc_id][segment_id]=[]
        score =  row['dp_runs_scores']
        systems[system][doc_id][segment_id].append(score)

    return systems


def load_da_scores_from_df(df, mqm):
    systems_comet_scores = {}
    systems_human_scores = {}
    systems_ext = {}
    print(df.columns)
    for i, row in df.iterrows():
        #print(row)
        if  mqm:
            sent_id = row['segment_id']
        else:
            sent_id = row['index']
        system = row['system']
        system_ext = 'system_'+system+'.json'
        doc_id = row['doc']
        if not system_ext in systems_ext:
            systems_ext[system_ext]=[]
        systems_ext[system_ext].append(doc_id)
        if not system_ext in systems_comet_scores:
            systems_comet_scores[system_ext]={}
        if not doc_id in systems_comet_scores[system_ext]:
            systems_comet_scores[system_ext][doc_id]={}
        if not system_ext in systems_human_scores:
            systems_human_scores[system_ext]={}
        if not doc_id in systems_human_scores[system_ext]:
            systems_human_scores[system_ext][doc_id]={}
        scores = row['dp_runs_scores']
        if not sent_id in systems_comet_scores[system_ext][doc_id]:
            systems_comet_scores[system_ext][doc_id][sent_id]=scores
        if not sent_id in systems_human_scores[system_ext][doc_id]:
            systems_human_scores[system_ext][doc_id][sent_id]=[]
        systems_human_scores[system_ext][doc_id][sent_id].extend([row['human_score']])
       
    return(systems_comet_scores, systems_human_scores, systems_ext)



def split_dev_test(comet_scores, scores, systems_list, dev_first):
    # split data into two sets, maintaining same docs across systems
    comet_scores_test = {}
    scores_test = {}
    comet_scores_dev = {}
    scores_dev = {}
    for system in systems_list:
        comet_sys_scores = comet_scores[system]
        sys_scores = scores[system]
        assert(len(comet_sys_scores)==len(sys_scores))
        if not system in comet_scores_test:
            comet_scores_test[system]={}
            scores_test[system]={}
            comet_scores_dev[system]={}
            scores_dev[system]={}
        # split based on the doc number
        dev_len = len(comet_sys_scores)//2
        
        for comet_doc_id, score_doc_id in zip(comet_sys_scores, sys_scores):
            assert(len(comet_sys_scores[comet_doc_id]) == len(sys_scores[score_doc_id]))
            if len(comet_scores_dev[system]) < dev_len:
                comet_scores_dev[system][comet_doc_id] = comet_sys_scores[comet_doc_id]
                scores_dev[system][score_doc_id]=sys_scores[score_doc_id]
            else:
                comet_scores_test[system][score_doc_id]=comet_sys_scores[score_doc_id]
                scores_test[system][score_doc_id]=sys_scores[score_doc_id]

    if dev_first:
        return(comet_scores_test, scores_test, comet_scores_dev,scores_dev)
    else:
        return(comet_scores_dev, scores_dev, comet_scores_test,scores_test)
        


def standardize(scores_test, scores_dev, norm):
    norm_mean = 0.0
    norm_std = 1.0
    if norm:
        norm_mean, norm_std = compute_z_norm_sent(scores_dev)
    all_scores = np.array([val for _,sys in scores_test.items() for _,doc in sys.items() for _,val in doc.items() ])
  
    all_scores -= norm_mean
    all_scores /= norm_std
 
    return all_scores, norm_mean, norm_std

def is_in_group(value, l):
    if value in l:
        return 1
    else:
        return 0 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process comet outputs')
    parser.add_argument('--comet-setup-file', type=str, 
                        help='path to comet setup to test on')
    parser.add_argument('--scores-file', type=str, 
                        help='path to scores for testing on')
    parser.add_argument('--norm', type=bool, default=True,
                        help='set to true to normalise the std on the ECE')
    parser.add_argument('--score-type', type=str, default='mqm', 
                        help='Choose type of scores between da | mqm')
    parser.add_argument('--dev-first', default=False, action='store_true',
                        help= 'select which half to be used as dev set')
    parser.add_argument('--cdf', type=int, default=-5,
                        help= 'value to calc sdf for')
    parser.add_argument('--optimise', default=False, action='store_true',
                        help= 'tune q selection on recall?')
    parser.add_argument('--lp', default='en-de', type=str,
                        help= 'select to evaluate the baseline only')
    parser.add_argument('--comet-original-file', type=str, default='',
                        help='path to comet original to test on')
    parser.add_argument('--prefix', type=str, default='new_')
    parser.add_argument('--norm_len', default=False, action='store_true')

    args = parser.parse_args()


    test_year='2020'
    if '2019' in args.comet_setup_file:
        test_year='2019'

    if args.score_type.lower()=='da':
        combined_df = get_df(args.comet_setup_file, args.scores_file, 100, args.docs, args.ens)
        systems_comet_scores, systems_human_scores, systems_ext = load_da_scores_from_df(combined_df, False)
    else:
        combined_df = get_dfs(args.comet_setup_file, args.scores_file, args.norm_len, 100, args.score_type, args.lp )
        print(combined_df.shape)
        systems_comet_scores, systems_human_scores, systems_ext = load_da_scores_from_df(combined_df, True)
    print(list(combined_df.columns))
    
    if args.comet_original_file!='':
       combined_df_origin = get_dfs(args.comet_original_file, args.scores_file, args.norm_len, 100, args.score_type, args.lp )
       systems_comet_scores_original, _, _ = load_da_scores_from_df(combined_df_origin, True)
 

    systems_comet_scores_test, systems_scores_test, systems_comet_scores_dev, systems_scores_dev = split_dev_test(
        systems_comet_scores, systems_human_scores, systems_ext, args.dev_first)
    systems_cometOr_scores_test, _, systems_cometOr_scores_dev, _ = split_dev_test(
        systems_comet_scores_original, systems_human_scores, systems_ext, args.dev_first)
    
    norm_human_test, mean_ht, std_ht = standardize(systems_scores_test, systems_scores_dev, args.norm)
    norm_comet_test, mean_ct, std_ct = standardize(systems_comet_scores_test, systems_comet_scores_dev, args.norm)
    norm_cometOr_test, mean_cot, std_cot = standardize(systems_cometOr_scores_test, systems_cometOr_scores_dev, args.norm)
    norm_comet_avg_test = norm_comet_test.mean(axis=1)
    norm_human_avg_test = norm_human_test.mean(axis=1)
    norm_cometOr_avg_test = norm_cometOr_test.mean(axis=1)

    # we need to repeat on the dev set to optimise the calibration parameters!
    norm_human_dev, mean_hd, std_hd = standardize(systems_scores_dev, systems_scores_dev, args.norm)
    norm_comet_dev, mean_cd, std_cd = standardize(systems_comet_scores_dev, systems_comet_scores_dev, args.norm)
    norm_cometOr_dev, mean_cod, std_cod = standardize(systems_cometOr_scores_dev, systems_cometOr_scores_dev, args.norm)
    norm_comet_avg_dev = norm_comet_dev.mean(axis=1)    
    norm_human_avg_dev = norm_human_dev.mean(axis=1)    
    norm_cometOr_avg_dev = norm_cometOr_dev.mean(axis=1)
    
    comet_std_dev = norm_comet_dev.std(axis=-1)
    comet_std_test = norm_comet_test.std(axis=-1)
    #calibrate std
    std_sums = np.linspace(0, 2, 100)
    std_scales = np.linspace(1, 10, 100)
    _, std_sum, std_scale, = optimize_calibration_error(
        norm_human_avg_dev, norm_comet_avg_dev, comet_std_dev,
        std_sums=std_sums, std_scales=std_scales)

    transformed_std = np.sqrt(std_sum**2 + (std_scale*comet_std_test)**2)
    transformed_std_dev = np.sqrt(std_sum**2 + (std_scale*comet_std_dev)**2)
    

    sample_index_text = np.linspace(0,len(norm_human_avg_test)-1, len(norm_human_avg_test)).astype(int)
    sample_index_dev = np.linspace(0,len(norm_human_avg_dev)-1, len(norm_human_avg_dev)).astype(int)
    
    norm_human_avg_test_sort,  sorted_index_test = (list(t) for t in zip(*sorted(zip(norm_human_avg_test, sample_index_text))))
    norm_human_avg_dev_sort,  sorted_index_dev = (list(t) for t in zip(*sorted(zip(norm_human_avg_dev, sample_index_dev))))
    norm_comet_avg_test_sort,  sorted_index_testc, sorted_std, sorted_std_pre = (list(t) for t in zip(*sorted(zip(norm_comet_avg_test, sample_index_text, transformed_std, comet_std_test))))
    norm_comet_avg_dev_sort,  sorted_index_devc, sorted_std_dev, sorted_std_dev_pre = (list(t) for t in zip(*sorted(zip(norm_comet_avg_dev, sample_index_dev, transformed_std_dev, comet_std_dev))))
    norm_cometOr_avg_test_sort,  sorted_index_Or_testc = (list(t) for t in zip(*sorted(zip(norm_cometOr_avg_test, sample_index_text))))
    norm_cometOr_avg_dev_sort,  sorted_index_Or_devc = (list(t) for t in zip(*sorted(zip(norm_cometOr_avg_dev, sample_index_dev))))
     
    
    print('test size %d' % (len(norm_human_avg_test_sort)))
    batches = [1, 2, 5, 10, 15, 20]
    
    ap_c = []
    ap_cu = []
    for i, batch in enumerate(batches):
        data_length = len(norm_human_avg_test)
        relevant_p = data_length*batch//100
        print(len(norm_human_avg_test))
        print(relevant_p)

        print('---------RELEVANT-----------')
        print(relevant_p)
       
        n_range=[]
        for i in range(5,len(norm_human_avg_test),5):
            n_range.append(i)
        
        
        precision_out_comet = []
        precision_out_comet_mcd = []
        precision_out_comet_uc = []

        
        recall_out_comet = []
        recall_out_comet_mcd = []
        recall_out_comet_uc = []
 
       
        
        data_length = len(norm_human_avg_test)
        relevant_p = data_length*batch//100
        print('---------RELEVANT-----------')
        print(relevant_p)

        q_range=np.linspace(-20, 20,  1001)
        if args.optimise:    
            opt_q = optimise_q(q_range, norm_human_avg_dev, norm_comet_avg_dev, transformed_std_dev, sample_index_dev, sorted_index_dev, mean_hd, std_hd, relevant_p)
        else:
            opt_q = args.cdf


        threshold = opt_q
        print(threshold)
        
        print('---------------')
        print('BIN %d - %f %d' % (i, threshold, relevant_p))
        comet_probs = calculate_score_cdf(norm_comet_avg_test, comet_std_test, threshold)
        comet_probs_cal = calculate_score_cdf(norm_comet_avg_test, transformed_std, threshold)
        original_probs = calculate_score_cdf(norm_cometOr_avg_test, comet_std_test, threshold)
        original_probs_cal = calculate_score_cdf(norm_cometOr_avg_test, transformed_std, threshold)
        
        srt_cometOr_probs, sorted_comet_original_truth =  (list(t) for t in zip(*sorted(zip(norm_cometOr_avg_test, sample_index_text))))
        srt_cometMCD_probs, sorted_cometMCD_truth =  (list(t) for t in zip(*sorted(zip(norm_comet_avg_test, sample_index_text))))
        srt_comet_probs, comet_std_cometsort_test, sorted_comet_truth = (list(t) for t in zip(*sorted(zip(comet_probs, comet_std_test, sample_index_text), reverse=True)))
        srt_comet_probs_cal, sorted_comet_cal_truth = (list(t) for t in zip(*sorted(zip(comet_probs_cal, sample_index_text), reverse=True))) 
         



        bin_target = sorted_index_test[:relevant_p]
       
        for prec in n_range:
            n=prec

            y_true = np.zeros(len(sorted_comet_original_truth))
            y_MCD = np.zeros(len(sorted_comet_original_truth)) 
            y_original = np.zeros(len(sorted_comet_original_truth))
            y_unc = np.zeros(len(sorted_comet_cal_truth))
            np.put(y_true,bin_target,1)
            np.put(y_MCD,sorted_cometMCD_truth[:n],1)
            np.put(y_original,sorted_comet_original_truth[:n],1)
            np.put(y_unc,sorted_comet_cal_truth[:n],1)
            ap_MCD = average_precision_score(y_true, y_MCD)
            ap_original = average_precision_score(y_true, y_original)
            ap_unc = average_precision_score(y_true, y_unc)

            prec_mcd = compute_precision(sorted_cometMCD_truth, bin_target, n)
            prec_base = compute_precision(sorted_comet_original_truth, bin_target, n)
            prec_comet_cal = compute_precision(sorted_comet_cal_truth, bin_target, n)
            
            precision_out_comet.append(prec_base)
            precision_out_comet_mcd.append(prec_mcd)
            precision_out_comet_uc.append(prec_comet_cal)
  
        for rec in n_range:
            n=rec
            rec_mcd = compute_recall(sorted_cometMCD_truth, bin_target, n)
            rec_base = compute_recall(sorted_comet_original_truth, bin_target, n)
            rec_comet_cal = compute_recall(sorted_comet_cal_truth, bin_target, n)
   
            recall_out_comet.append(rec_base)
            recall_out_comet_mcd.append(rec_mcd)
            recall_out_comet_uc.append(rec_comet_cal)

            
        
        n=relevant_p

        ap_c.append(compute_ap(sorted_cometMCD_truth, sorted_index_test, n))
        ap_cu.append(compute_ap(sorted_comet_truth, sorted_index_test, n))

        matplotlib.rc('xtick', labelsize=14) 
        matplotlib.rc('ytick', labelsize=14) 
        plt.figure(figsize=(6.5,2))
        plt.xlabel('N', fontsize=14)
        plt.ylabel('Precision@N', fontsize=14)
        plt.plot(n_range, precision_out_comet, 'royalblue', label="(1) COMET original")
        plt.plot(n_range, precision_out_comet_mcd,  'orangered', linestyle='dashed', label="(2) MCD COMET mean")
        plt.plot(n_range, precision_out_comet_uc, 'darkgreen',linestyle='dotted', label="(3) UA-COMET")

        
        plt.legend()
        plt.legend(prop={'size': 13})
        plt.show()
        plt.savefig('figures/'+args.prefix+args.score_type.upper()+'-Precision@N_relevant_'+str(batch)+'_perc.png',bbox_inches = "tight")
        plt.close()

        matplotlib.rc('xtick', labelsize=14) 
        matplotlib.rc('ytick', labelsize=14) 
        plt.figure(figsize=(6.5,2))
        plt.xlabel('N', fontsize=14)
        plt.ylabel('Recall@N', fontsize=14)
        plt.plot(n_range, recall_out_comet, 'royalblue', label="(1) COMET original")
        plt.plot(n_range, recall_out_comet_mcd, 'orangered', linestyle='dashed', label="(2) MCD COMET mean")
        plt.plot(n_range, recall_out_comet_uc, 'darkgreen',linestyle='dotted', label="(3) UA-COMET")
       
  
        plt.legend()
        plt.legend(prop={'size': 13})
        plt.show()
        plt.savefig('figures/'+args.prefix+args.score_type.upper()+'-Recall@N_relevant_'+str(batch)+'_perc.png',bbox_inches = "tight")
        plt.close()