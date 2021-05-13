import json
import csv
import numpy as np
import matplotlib.pyplot as plt
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
from error_ir import *


def get_df(comet_dir, da_dir, nruns=100, docs=False, ens=True):
  
    SETUP_PATH = comet_dir
    files = [f for f in listdir(SETUP_PATH) if isfile(join(SETUP_PATH, f))]
    sys_files = [f for f in files if f.split('_')[0] == 'system']

    da_scores = pd.read_csv(da_dir)
    da_scores.system = da_scores.system.apply(lambda x: x.split('.')[0])

    dfs = []

    for s in sys_files:
        f = open(join(SETUP_PATH,s), 'r')
        data = json.loads(f.read()) 
        f.close() 
        #print(len(data))
        system_name = '_'.join(s.split('.')[0].split('_')[1:])
        lines = [[i['src'], i['mt'], i['ref'], i['dp_runs_scores']] for i in data if 'dp_runs_scores' in i.keys()]
        df_ = pd.DataFrame(data=np.array(lines), columns=['src','mt', 'ref', 'dp_runs_scores'])
        da_scores_ = da_scores[da_scores.system == system_name]
        df = df_.merge(da_scores_, how='inner', on=['src', 'mt'])
        #print(len(df))
        
        df['dp_runs_scores'] = df['dp_runs_scores'].apply(lambda x: x[:nruns])
        df['predicted_score_mean'] = df['dp_runs_scores'].apply(lambda x: np.mean(x)) # segment-level
        df['predicted_score_std'] = df['dp_runs_scores'].apply(lambda x: np.std(x)) # segment-level
        df['q-mu'] = np.abs(df['human_score'] - df['predicted_score_mean'])
        #df.drop(['ref_x', 'ref_y', 'lp', 'annotators'], axis=1, inplace=True)
        dfs.append(df)
    df_all = pd.concat(dfs)
    df_all.reset_index(inplace=True)
    #df_all.dp_runs_scores = df_all.dp_runs_scores.apply(lambda x: [float(i) for i in x.split('|')])        

   
    return df_all


def get_dfs(comet_dir, score_file, nruns, docs=False, t='mqm', ens=True):
    mqms_ub = pd.read_parquet(score_file, engine='fastparquet') ##requires installation of fastparquet
    
    SETUP_PATH = comet_dir
    files = [f for f in listdir(SETUP_PATH) if isfile(join(SETUP_PATH, f))]
    sys_files = [f for f in files if f.split('_')[0] == 'system']

    dfs=[]
    mqms_ub.system = mqms_ub.system.apply(lambda x: x.split('.')[0])
    #print(len(mqms_ub))
    nruns=100
    for s in sys_files:
        f = open(join(SETUP_PATH,s), 'r')
        data = json.loads(f.read()) 
        f.close() 
        #print(len(data))
        system_name = '_'.join(s.split('.')[0].split('_')[1:])
        lines = [[i['src'], i['mt'], i['ref'], i['dp_runs_scores']] for i in data if 'dp_runs_scores' in i.keys()]
        df_ = pd.DataFrame(data=np.array(lines), columns=['source_segment','target_segment', 'ref', 'dp_runs_scores'])
        #print(len(df_))
        mqms_ub_ = mqms_ub[mqms_ub.system == system_name]
        #print(len(mqms_ub_))
        df = df_.merge(mqms_ub_, how='inner', on=['source_segment', 'target_segment'])
        #print(len(df))
        df['dp_runs_scores'] = df['dp_runs_scores'].apply(lambda x: x[:nruns])
        df['predicted_score_mean'] = df['dp_runs_scores'].apply(lambda x: np.mean(x)) # segment-level
        df['predicted_score_std'] = df['dp_runs_scores'].apply(lambda x: np.std(x)) # segment-level
        df['q-mu'] = np.abs(df['target_segment_mqm'] - df['predicted_score_mean'])
        #df.drop(['ref_x', 'ref_y', 'lp', 'annotators'], axis=1, inplace=True)
        dfs.append(df)
        
    df_all = pd.concat(dfs)
    df_all.reset_index(inplace=True)
    #df_all.dp_runs_scores = df_all.dp_runs_scores.apply(lambda x: [float(i) for i in x.split('|')])
    #print('----df size----')
    #print(len(df_all))
    if t=='mqm':
        
        df_all.rename(columns={'target_segment_mqm':'human_score'}, inplace=True)

    return df_all



def get_system_scores_human(df):
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
        sent_id = row['index']
        if  mqm:
            sent_id = row['segment_id']
        system = row['system']
        system_ext = 'system_'+system+'.json'
        doc_id = row['doc_id']
        if not system_ext in systems_ext:
            #systems_ext.append(system_ext)
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
        #scores = [float(i) for i in row['dp_runs_scores']]
        scores = row['dp_runs_scores']
        if not sent_id in systems_comet_scores[system_ext][doc_id]:
            systems_comet_scores[system_ext][doc_id][sent_id]=scores
        if not sent_id in systems_human_scores[system_ext][doc_id]:
            systems_human_scores[system_ext][doc_id][sent_id]=[]
        #if (len(systems_human_scores[system_ext][doc_id][sent_id]))==0:
        systems_human_scores[system_ext][doc_id][sent_id].extend([row['human_score']])
        #else:
        #    systems_human_scores[system_ext][doc_id][sent_id].extend(row['human_score'])
        
        #print(systems_human_scores[system_ext][doc_id][sent_id])   
    return(systems_comet_scores, systems_human_scores, systems_ext)


def split_dev_test(comet_scores, scores, systems_list, dev_first):
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
        count = 0
        
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
    parser.add_argument('--docs', default=False, action='store_true',
                        help= 'select segment or document level eval')
    parser.add_argument('--cdf', type=int, default=-5,
                        help= 'value to calc sdf for')
    parser.add_argument('--opt', default=False, action='store_true',
                        help= 'tune q selection on recall?')
    parser.add_argument('--baseline', default=False, action='store_true',
                        help= 'select to evaluate the baseline only')
    parser.add_argument('--ens', default=False, action='store_true',
                        help= 'select to evaluate the baseline only')

    args = parser.parse_args()


    test_year='2020'
    if '2019' in args.comet_setup_file:
        test_year='2019'

    if args.score_type.lower()=='da':
        combined_df = get_df(args.comet_setup_file, args.scores_file, 100, args.docs, args.ens)
        systems_comet_scores, systems_human_scores, systems_ext = load_da_scores_from_df(combined_df, False)
    else:
        combined_df = get_dfs(args.comet_setup_file, args.scores_file, 100, args.docs, args.score_type, args.ens)
        systems_comet_scores, systems_human_scores, systems_ext = load_da_scores_from_df(combined_df, True)
    print(list(combined_df.columns))

    
    systems_comet_scores_test, systems_scores_test, systems_comet_scores_dev, systems_scores_dev = split_dev_test(
        systems_comet_scores, systems_human_scores, systems_ext, args.dev_first)

    norm_human_test, mean_ht, std_ht = standardize(systems_scores_test, systems_scores_dev, args.norm)
    norm_comet_test, mean_ct, std_ct = standardize(systems_comet_scores_test, systems_comet_scores_dev, args.norm)
    norm_comet_avg_test = norm_comet_test.mean(axis=1)
    norm_human_avg_test = norm_human_test.mean(axis=1)

    # we need to repeat on the dev set to optimise the calibration parameters!
    norm_human_dev, mean_hd, std_hd = standardize(systems_scores_dev, systems_scores_dev, args.norm)
    norm_comet_dev, mean_cd, std_cd = standardize(systems_comet_scores_dev, systems_comet_scores_dev, args.norm)
    norm_comet_avg_dev = norm_comet_dev.mean(axis=1)    
    norm_human_avg_dev = norm_human_dev.mean(axis=1)    

    print(norm_human_avg_test[:10])
    print(norm_comet_avg_test[:10])
    comet_std_dev = norm_comet_dev.std(axis=-1)
    comet_std_test = norm_comet_test.std(axis=-1)
    std_sums = np.linspace(0, 2, 100)
    std_scales = np.linspace(1, 10, 100)
    _, std_sum, std_scale, = optimize_calibration_error(
        norm_human_avg_dev, norm_comet_avg_dev, comet_std_dev,
        std_sums=std_sums, std_scales=std_scales)

    transformed_std = np.sqrt(std_sum**2 + (std_scale*comet_std_test)**2)
    transformed_std_dev = np.sqrt(std_sum**2 + (std_scale*comet_std_dev)**2)

    #print(norm_comet_avg_test[0].shape)

    fixed_std =  compute_fixed_std(norm_comet_avg_dev, norm_human_avg_dev)
    baseline_stds_test = np.full_like(comet_std_test, fixed_std)
    baseline_stds_dev = np.full_like(comet_std_dev, fixed_std)
    sample_index_text = np.linspace(0,len(norm_human_avg_test)-1, len(norm_human_avg_test)).astype(int)
    sample_index_dev = np.linspace(0,len(norm_human_avg_dev)-1, len(norm_human_avg_dev)).astype(int)
    norm_human_avg_test_sort,  sorted_index_test = (list(t) for t in zip(*sorted(zip(norm_human_avg_test, sample_index_text))))
    norm_human_avg_dev_sort,  sorted_index_dev = (list(t) for t in zip(*sorted(zip(norm_human_avg_dev, sample_index_dev))))
    print('test size %d' % (len(norm_human_avg_test_sort)))
    batches = [1, 10]

    precision = [5, 10, 15, 20, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    recall    = [5, 10, 15, 20, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    ap_c = []
    ap_cu = []
    for i, batch in enumerate(batches):
        data_length = len(norm_human_avg_test)
        relevant_p = data_length*batch//100
        print('---------RELEVANT-----------')
        print(relevant_p)
        precision = [5, 10, 15, 20, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
        recall    = [5, 10, 15, 20, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
        batch_precision=[]
        for pi,pr in enumerate(precision):
            if pr <= relevant_p:
                batch_precision.append(pr)
        batch_precision.append(relevant_p)
        tot_precision=[]
        for pi,pr in enumerate(precision):
            if pr <= relevant_p:
                tot_precision.append(pr)
        tot_precision.append(len(norm_human_avg_test))
        batch_recall=[]
        for pi,pr in enumerate(recall):
            if pr <= relevant_p:
                batch_recall.append(pr)
        batch_recall.append(relevant_p)
        tot_recall = []
        for pi,pr in enumerate(recall):
            if pr <= relevant_p:
                tot_recall.append(pr)
        tot_recall.append(len(norm_human_avg_test))
        prec_out_comet = []
        prec_out_comet_u = []
        prec_out_comet_b = []
        prec_out_comet_uc = []
        rec_out_comet = []
        rec_out_comet_u = []
        rec_out_comet_b = []
        rec_out_comet_uc = []
        prect_out_comet = []
        prect_out_comet_u = []
        prect_out_comet_b = []
        prect_out_comet_uc = []
        rect_out_comet = []
        rect_out_comet_u = []
        rect_out_comet_b = []
        rect_out_comet_uc = []
        oprect_out_comet = []
        oprect_out_comet_u = []
        oprect_out_comet_b = []
        oprect_out_comet_uc = []
        orect_out_comet = []
        orect_out_comet_u = []
        orect_out_comet_b = []
        orect_out_comet_uc = []
        
        data_length = len(norm_human_avg_test)
        relevant_p = data_length*batch//100
        print('---------RELEVANT-----------')
        print(relevant_p)
        target_threshold = norm_human_avg_test_sort
        q_range=np.linspace(-20, 20,  401)
        if args.opt:
            #opt_q = optimise_q(q_range, norm_human_avg_dev, norm_comet_avg_dev, transformed_std_dev, sample_index_dev, sorted_index_dev, mean_hd, std_hd, relevant_p)
            opt_q = optimise_q(q_range, norm_human_avg_dev, norm_comet_avg_dev, comet_std_dev, sample_index_dev, sorted_index_dev, mean_hd, std_hd, relevant_p)
        else:
            opt_q = args.cdf

        threshold = (opt_q-mean_ht)/std_ht
        print(threshold)
        
        print('---------------')
        print('BIN %d - %f -%d' % (i, threshold, relevant_p))
        base_probs = calculate_score_cdf(norm_comet_avg_test, baseline_stds_test, threshold)
        comet_probs = calculate_score_cdf(norm_comet_avg_test, comet_std_test, threshold)
        comet_probs_cal = calculate_score_cdf(norm_comet_avg_test, transformed_std, threshold)


        srt_base1_probs, comet_std_basesort_test, sorted_base1_truth =  (list(t) for t in zip(*sorted(zip(norm_comet_avg_test, comet_std_test, sample_index_text))))
        srt_base_probs, sorted_base_truth  = (list(t) for t in zip(*sorted(zip(base_probs,  sample_index_text), reverse=True)))
        srt_comet_probs, comet_std_cometsort_test, sorted_comet_truth = (list(t) for t in zip(*sorted(zip(comet_probs, comet_std_test, sample_index_text), reverse=True)))
        srt_comet_probs_cal, sorted_comet_cal_truth = (list(t) for t in zip(*sorted(zip(comet_probs_cal, sample_index_text), reverse=True))) 
        print(norm_human_avg_test_sort[:100])
        print(sorted_index_test[:100])
        print(sorted_base1_truth[:100])
        print(comet_std_basesort_test[:100])
        print(srt_base1_probs[:100])
        print(sorted_base_truth[:100])
        print(srt_base_probs[:100])
        print(sorted_comet_truth[:100])
        print(srt_comet_probs[:100])
        print(comet_std_cometsort_test[:100])
        print(sorted_comet_cal_truth[:100])
        print(srt_comet_probs_cal[:100])


        bin_target = sorted_index_test[:relevant_p]
            
        for prec in batch_precision:
            n=prec
            prec_base1 = compute_precision(sorted_base1_truth, bin_target, n)
            prec_base = compute_precision(sorted_base_truth, bin_target, n)
            prec_comet = compute_precision(sorted_comet_truth, bin_target, n)
            prec_comet_cal = compute_precision(sorted_comet_cal_truth, bin_target, n)
      
            prec_out_comet.append(prec_base1)
            prec_out_comet_u.append(prec_comet)
            prec_out_comet_b.append(prec_base)
            prec_out_comet_uc.append(prec_comet_cal)

        for rec in batch_recall:
            n=rec
            rec_base1 = compute_recall(sorted_base1_truth, bin_target, n)
            rec_base = compute_recall(sorted_base_truth, bin_target, n)
            rec_comet = compute_recall(sorted_comet_truth, bin_target, n)
            rec_comet_cal = compute_recall(sorted_comet_cal_truth, bin_target, n)
       
            rec_out_comet.append(rec_base1)
            rec_out_comet_u.append(rec_comet)
            rec_out_comet_b.append(rec_base)
            rec_out_comet_uc.append(rec_comet_cal)

        for prec in tot_precision:
            n=prec
            prec_base1 = compute_precision(sorted_base1_truth, bin_target, n)
            prec_base = compute_precision(sorted_base_truth, bin_target, n)
            prec_comet = compute_precision(sorted_comet_truth, bin_target, n)
            prec_comet_cal = compute_precision(sorted_comet_cal_truth, bin_target, n)
            
            prect_out_comet.append(prec_base1)
            prect_out_comet_u.append(prec_comet)
            prect_out_comet_b.append(prec_base)
            prect_out_comet_uc.append(prec_comet_cal)

        for rec in tot_recall:
            n=rec
            rec_base1 = compute_recall(sorted_base1_truth, bin_target, n)
            rec_base = compute_recall(sorted_base_truth, bin_target, n)
            rec_comet = compute_recall(sorted_comet_truth, bin_target, n)
            rec_comet_cal = compute_recall(sorted_comet_cal_truth, bin_target, n)
        
            rect_out_comet.append(rec_base1)
            rect_out_comet_u.append(rec_comet)
            rect_out_comet_b.append(rec_base)
            rect_out_comet_uc.append(rec_comet_cal)

        for prec in precision:
            n=prec
            prec_base1 = compute_precision(sorted_base1_truth, bin_target, n)
            prec_base = compute_precision(sorted_base_truth, bin_target, n)
            prec_comet = compute_precision(sorted_comet_truth, bin_target, n)
            prec_comet_cal = compute_precision(sorted_comet_cal_truth, bin_target, n)
            
            oprect_out_comet.append(prec_base1)
            oprect_out_comet_u.append(prec_comet)
            oprect_out_comet_b.append(prec_base)
            oprect_out_comet_uc.append(prec_comet_cal)

        for rec in recall:
            n=rec
            rec_base1 = compute_recall(sorted_base1_truth, bin_target, n)
            rec_base = compute_recall(sorted_base_truth, bin_target, n)
            rec_comet = compute_recall(sorted_comet_truth, bin_target, n)
            rec_comet_cal = compute_recall(sorted_comet_cal_truth, bin_target, n)
        
            orect_out_comet.append(rec_base1)
            orect_out_comet_u.append(rec_comet)
            orect_out_comet_b.append(rec_base)
            orect_out_comet_uc.append(rec_comet_cal)




        #latex friendly
        n=relevant_p

        ap_c.append(compute_ap(sorted_base1_truth, sorted_index_test, n))
        ap_cu.append(compute_ap(sorted_comet_truth, sorted_index_test, n))

        print('---------------------save-------------------------------')
        #print(prec_out_comet)
        #print(prec_out_comet_u)
        print(oprect_out_comet)
        print(oprect_out_comet_u)
        #print(rec_out_comet)
        #print(rec_out_comet_u)
        print(orect_out_comet)
        print(orect_out_comet_u)
        print('---------------------save-------------------------------')


        plt.xlabel(' N ')
        plt.ylabel('Precision')
        plt.title('Precision @ N - # errors = %d, (%d %%) ' % (relevant_p, batch))
        plt.plot(batch_precision, prec_out_comet, 'b', label="Original COMET")
        plt.plot(batch_precision, prec_out_comet_u, 'r', label="Uncertainty-aware COMET")
     #   plt.plot(batch_precision, prec_out_comet_b, 'b:', label="Original COMET + fixed std")
     #   plt.plot(batch_precision, prec_out_comet_uc, 'r:', label="Uncertainty-aware COMET + calibration")
        
        plt.legend()
        plt.show()
        plt.savefig('figures/'+args.score_type.upper()+'_wmt2020-'+test_year+'-Precision@N_relevant_'+str(batch)+'_perc.png')
        plt.close()

        plt.xlabel(' N ')
        plt.ylabel('Recall')
        plt.title('Recall @ N - # errors = %d, (%d %%) ' % (relevant_p, batch))
        plt.plot(batch_recall, rec_out_comet, 'b', label="Original COMET")
        plt.plot(batch_recall, rec_out_comet_u, 'r', label="Uncertainty-aware COMET")
        #plt.plot(batch_recall, rec_out_comet_b, 'b:', label="Original COMET + fixed std")
        #plt.plot(batch_recall, rec_out_comet_uc, 'r:', label="Uncertainty-aware COMET + calibration")
        
        plt.legend()
        plt.show()
        plt.savefig('figures/'+args.score_type.upper()+'_wmt2020-'+test_year+'-Recall@N_relevant_'+str(batch)+'_perc.png')
        plt.close()

        #############TOT

        plt.xlabel(' N ')
        plt.ylabel('Precision')
        plt.title('Precision @ N - # errors = %d, (%d %%) ' % (relevant_p, batch))
        plt.plot(tot_precision, prect_out_comet, 'b', label="Original COMET")
        plt.plot(tot_precision, prect_out_comet_u, 'r', label="Uncertainty-aware COMET")
     #   plt.plot(batch_precision, prec_out_comet_b, 'b:', label="Original COMET + fixed std")
     #   plt.plot(batch_precision, prec_out_comet_uc, 'r:', label="Uncertainty-aware COMET + calibration")
        
        plt.legend()
        plt.show()
        plt.savefig('figures/'+args.score_type.upper()+'_wmt2020-'+test_year+'-totPrecision@N_relevant_'+str(batch)+'_perc.png')
        plt.close()

        plt.xlabel(' N ')
        plt.ylabel('Recall')
        plt.title('Recall @ N - # errors = %d, (%d %%) ' % (relevant_p, batch))
        plt.plot(tot_recall, rect_out_comet, 'b', label="Original COMET")
        plt.plot(tot_recall, rect_out_comet_u, 'r', label="Uncertainty-aware COMET")
        #plt.plot(batch_recall, rec_out_comet_b, 'b:', label="Original COMET + fixed std")
        #plt.plot(batch_recall, rec_out_comet_uc, 'r:', label="Uncertainty-aware COMET + calibration")
        
        plt.legend()
        plt.show()
        plt.savefig('figures/'+args.score_type.upper()+'_wmt2020-'+test_year+'-totRecall@N_relevant_'+str(batch)+'_perc.png')
        plt.close()

        plt.xlabel(' N ')
        plt.ylabel('Precision')
        plt.title('Precision @ N - # errors = %d, (%d %%) ' % (relevant_p, batch))
        plt.plot(precision, oprect_out_comet, 'b', label="Original COMET")
        plt.plot(precision, oprect_out_comet_u, 'r', label="Uncertainty-aware COMET")
     #   plt.plot(batch_precision, prec_out_comet_b, 'b:', label="Original COMET + fixed std")
     #   plt.plot(batch_precision, prec_out_comet_uc, 'r:', label="Uncertainty-aware COMET + calibration")
        
        plt.legend()
        plt.show()
        plt.savefig('figures/'+args.score_type.upper()+'_wmt2020-'+test_year+'-ototPrecision@N_relevant_'+str(batch)+'_perc.png')
        plt.close()

        plt.xlabel(' N ')
        plt.ylabel('Recall')
        plt.title('Recall @ N - # errors = %d, (%d %%) ' % (relevant_p, batch))
        plt.plot(recall, orect_out_comet, 'b', label="Original COMET")
        plt.plot(recall, orect_out_comet_u, 'r', label="Uncertainty-aware COMET")
        #plt.plot(batch_recall, rec_out_comet_b, 'b:', label="Original COMET + fixed std")
        #plt.plot(batch_recall, rec_out_comet_uc, 'r:', label="Uncertainty-aware COMET + calibration")
        
        plt.legend()
        plt.show()
        plt.savefig('figures/'+args.score_type.upper()+'_wmt2020-'+test_year+'-ototRecall@N_relevant_'+str(batch)+'_perc.png')
        plt.close()

    print(ap_c)
    print(ap_cu)







    