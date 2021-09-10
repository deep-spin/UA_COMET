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
from os.path import isfile, join, isdir
from normalisation import compute_z_norm, compute_fixed_std, standardize
import copy
import random



def get_df_mqm(scores_file_g, lp):
   
    
    mqm_scores = pd.read_csv(scores_file_g)
    mqm_scores = mqm_scores[mqm_scores['lp']==lp]
    mqm_scores.system = mqm_scores.system.apply(lambda x: x.split('.')[0])
    mqm_scores =mqm_scores.rename(columns={'score':'human_score'})
    mqm_scores.loc[:, 'human_score'] = mqm_scores['human_score'].apply(lambda x: 100-x) #reverse scoring order
    return mqm_scores


def get_score_multi_ref(row, num, sample):
    scores = []
    if sample:
        ref_num = random.randint(0, num-1)
        return row['dp_runs_scores_'+str(ref_num)]
    else:
        for ref in range(num):
            scores.append(row['dp_runs_scores_'+str(ref)])
        scores_mean = np.mean(scores, axis=0)
        #print(scores_mean.shape)
        return scores_mean.tolist()


def combine_both_multi(comet_dir, g_df, nruns):
    
    SETUP_PATH = comet_dir
    #read data from the ub mqm scores as well to map the document ids
    #mqms_ub = pd.read_parquet(score_file, engine='fastparquet') ##requires installation of fastparquet
    #mqms_ub.system = mqms_ub.system.apply(lambda x: x.split('.')[0])
    
    #da_scores = mqms_ub
    #da_scores_ = da_scores.groupby(['source_segment', 'target_segment',  'doc_id', 'system'], as_index=False).mean()
    #print(da_scores_.shape)
    print(g_df.shape)
    #da_scores = da_scores_.rename(columns={'source_segment': 'src', 'target_segment':'mt'})
    
    #drop the ub scores, we only use the original google scores
    #da_scores.drop(['target_segment_mqm'], axis=1, inplace=True)
    #merge on src, mt and system name so that we get the doc ids too
    da_scores = g_df #da_scores.merge(g_df, how='inner', on=['src', 'mt', 'system'])
    #print(da_scores.shape)
    #print(da_scores.system.unique())


    # read the forward run outputs of the model (COMET)
    # expects 1 dir with subdirs for each reference
    # expects each ref dir to have 1 file per MT system
    df_all = []
    refs_dirs = [f for f in listdir(SETUP_PATH) if isdir(join(SETUP_PATH, f))]
    for di, dir in enumerate(refs_dirs):
        print(dir)
        dpath = join(SETUP_PATH, dir)
        print(dpath)
        files = [f for f in listdir(dpath) if isfile(join(dpath, f))]
        sys_files = [f for f in files if (f.split('_')[0] == 'system') and ('Human' not in f)]
       
        dfs = [] #reset for each reference
        # di=0 first reference, initialise rows in the df; the rest references will extend these rows
        if di==0:
            for s in sys_files:
                f = open(join(dpath,s), 'r')
                data = json.loads(f.read()) 
                f.close() 
                system_name = '_'.join(s.split('.')[0].split('_')[1:])   
                lines = [[i['src'], i['mt'], i['ref'], i['dp_runs_scores']] for i in data if 'dp_runs_scores' in i.keys()]
                df_ = pd.DataFrame(data=np.array(lines), columns=['src','mt', 'ref', 'dp_runs_scores'])
                da_scores_ = da_scores[da_scores.system == system_name]
                print
                print(system_name)
                print(da_scores_.shape)
                df = df_.merge(da_scores_, how='inner', on=['src', 'mt'])
                df['dp_runs_scores'] = df['dp_runs_scores'].apply(lambda x: x[:nruns])
                df = df.rename(columns={'ref': 'ref_'+str(di), 'dp_runs_scores':'dp_runs_scores_'+str(di)})
                #df.drop(['ref_x', 'ref_y'], axis=1, inplace=True)
                dfs.append(df)
            df_all = pd.concat(dfs)
            df_all.reset_index(inplace=True)
            #print(di)
            #print(df_all.shape)
        else:
            print(sys_files)
            for s in sys_files:
                f = open(join(dpath,s), 'r')
                data = json.loads(f.read()) 
                f.close() 
                system_name = '_'.join(s.split('.')[0].split('_')[1:])
                #gets previous version and expands
                df_all_s = df_all[df_all.system == system_name]
                
                lines = [[i['src'], i['mt'], i['ref'], i['dp_runs_scores']] for i in data if 'dp_runs_scores' in i.keys()]
                df_ = pd.DataFrame(data=np.array(lines), columns=['src','mt', 'ref', 'dp_runs_scores'])
                df_['dp_runs_scores'] = df_['dp_runs_scores'].apply(lambda x: x[:nruns])
                df_ = df_.rename(columns={'ref': 'ref_'+str(di), 'dp_runs_scores':'dp_runs_scores_'+str(di)})
                df = df_.merge(df_all_s, how='inner', on=['src', 'mt'])
                dfs.append(df)
            df_all = pd.concat(dfs)
            print(di)
            print(df_all.shape)
                    
    print(df_all.shape)
    #print(df_all.head())
    return df_all



def get_score_multi_ref_subsample(row, num, sample):
    scores = []
    # get all prossible combinations
    refpairs = list(itertools.combinations(list(range(num)), 2))
    # randomly pick pair
    ref_num = random.randint(0, len(refpairs)-1)
    pair = refpairs[ref_num]
    #print(pair)
    sample_int = random.randint(0,1)
    scores_sampled = row['dp_runs_scores_'+str(pair[sample_int])]
    for ref in pair:
        scores.append(row['dp_runs_scores_'+str(ref)])
    scores_mean = np.mean(scores, axis=0)
        
    if sample:
        return scores_sampled.tolist()
    else:
        return scores_mean.tolist()
    
    return row['dp_runs_scores_'+str(ref_num)]

    return scores_mean.tolist(), scores_sampled.tolist()


def load_da_scores_from_df_multi(df, num_ref, sample, paireval):
    systems_comet_scores = {}
    systems_human_scores = {}
    systems_ext = {}
    for i, row in df.iterrows():
        sent_id = int(row['index'])
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
            systems_comet_scores[system_ext][doc_id]=[]
        if not system_ext in systems_human_scores:
            systems_human_scores[system_ext]={}
        if not doc_id in systems_human_scores[system_ext]:
            systems_human_scores[system_ext][doc_id]=[]
        if paireval:
            scores = get_score_multi_ref_subsample(row, num_ref, sample)
        else:
            scores = get_score_multi_ref(row, num_ref, sample)
        systems_comet_scores[system_ext][doc_id].append(scores)
        systems_human_scores[system_ext][doc_id].append(row['human_score'])
        
    return(systems_comet_scores, systems_human_scores, systems_ext)


def split_k_fold(comet_scores, scores, systems_list, k=5):
    final_folds = []
    for system in systems_list:
        comet_sys_scores = comet_scores[system]
        sys_scores = scores[system]
        assert(len(comet_sys_scores) == len(sys_scores))
        # split based on the doc level
        zipped = [i for i in zip(comet_sys_scores, sys_scores)]
        folds = np.array_split(zipped, k)
        for i, fold in enumerate(folds):
            comet_scores_fold = {}
            human_scores_fold = {}
            if system not in comet_scores_fold:
                comet_scores_fold[system] = {}
                human_scores_fold[system] = {}
            for comet_doc_id, score_doc_id in fold:
                assert(len(comet_sys_scores[comet_doc_id]) == len(sys_scores[score_doc_id]))
                comet_scores_fold[system][comet_doc_id] = comet_sys_scores[comet_doc_id]
                human_scores_fold[system][score_doc_id] = sys_scores[score_doc_id]

            if len(final_folds) <= i:
                final_folds.append({})
                final_folds[i]['human'] = {}
                final_folds[i]['comet'] = {}

            final_folds[i]['human'][system] = human_scores_fold[system]
            final_folds[i]['comet'][system] = comet_scores_fold[system]
    
    # levels of nested dicts : 
    # 5 folds -> each fold has 2 dicts 'human' and "comet" ->
    # -> each 'human' and "comet" has N keys == systems -> docs -> segments
    return final_folds 


def merge_folds(list_of_folds):
    dev_fold = {}
    for i, fold in enumerate(list_of_folds):
        if i == 0:
            dev_fold = copy.deepcopy(fold)
        else:
            for human_sys, comet_sys in zip(fold['human'].keys(), fold['comet'].keys()):
                dev_fold['human'][human_sys].update(fold['human'][human_sys])
                dev_fold['comet'][comet_sys].update(fold['comet'][comet_sys])
    return dev_fold
        

def batch_data(all_da, all_comet, all_comet_avg, batch_size=1):
    n = len(all_da) - (len(all_da) % batch_size)
    batch_da = all_da[:n].reshape(n//batch_size, batch_size).mean(axis=1)
    #print(all_comet[:n, :].reshape(n//batch_size, batch_size, -1).shape)
    batch_comet_scores = [i.mean(axis=0) for i in all_comet[:n, :].reshape(n//batch_size, batch_size, -1)]
    batch_comet_avg = all_comet_avg[:n].reshape(
        n//batch_size, batch_size).mean(axis=1)
    batch_comet_std = all_comet[:n, :].reshape(
        n//batch_size, batch_size, -1).mean(axis=1).std(axis=-1)
    return batch_da, batch_comet_scores, batch_comet_avg, batch_comet_std



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process comet outputs')
    parser.add_argument('--comet-setup-file', type=str, 
                        help='path to comet setup to test on')
    parser.add_argument('--scores-file', type=str, 
                        help='path to scores for testing on')
    parser.add_argument('--norm', type=bool, default=True,
                        help='set to true to normalise the std on the ECE')
    parser.add_argument('--score-type', type=str, default='mqm', 
                        help='Choose type of scores between da | mqm | hter')
    parser.add_argument('--nruns', type=int, default=100,
                        help= 'select how many drpout runs to evaluate')
    parser.add_argument('--baseline', default=False, action='store_true',
                        help= 'select to evaluate the baseline only')
    parser.add_argument('--sample', default=False, action='store_true',
                        help= 'if set it will sample through multiple references instead of averaging them')
    parser.add_argument('--numrefs',type=int, default=3,
                        help= 'select how many refs to evaluate')
    parser.add_argument('--doc-annotations',type=str, default='',
                        help= 'for mqm')
    parser.add_argument('--paireval', default=False, action='store_true',
                        help= 'if set it will sample through pairs of references ')
    parser.add_argument('--lp', type=str, default='en-de')
    
    args = parser.parse_args()
    
    test_year='2020'
    if '2019' in args.comet_setup_file:
        test_year='2019'
    
    random.seed(10)
    
    #load MQM annotations merge
    df_all_g = get_df_mqm(args.scores_file, args.lp)
    print(df_all_g.columns)
    print(df_all_g.shape)
    # merge with document level annotations to be able to maintain whole documents when splitting
    df_combined = combine_both_multi(args.comet_setup_file, df_all_g, args.nruns)
    print(df_combined.shape)
    # break to comet vs human scores for processing
    systems_comet_scores, systems_human_scores, systems_ext = load_da_scores_from_df_multi(df_combined, args.numrefs, args.sample, args.paireval)
    k_folds = split_k_fold(systems_comet_scores, systems_human_scores, systems_ext)

    cal_avgll_folds = []
    calibration_error_folds = []
    sharpness_cal_folds = []
    pearson_acc_folds = []
    pearson_d1_cal_folds = []

    for i, fold in enumerate(k_folds):
        keys = np.arange(len(k_folds))  # 0,1,2,3,4
        dev = keys[keys != i]
        merged_dev = merge_folds([k_folds[k] for k in dev])
        # systems_comet_scores_test, systems_scores_test = k_folds[i]['comet'], k_folds[i]['human']
        systems_comet_scores_test, systems_scores_test = fold['comet'], fold['human']
        systems_comet_scores_dev, systems_scores_dev = merged_dev['comet'], merged_dev['human']
        

        print()
        print('- - - making dev/test split - - -')
        print('processing as test fold #', i)
        print('processing as dev folds #', dev)

        print()
        print(len(systems_comet_scores_test))
        print(len(systems_scores_test))
        
        norm_human_test = standardize(systems_scores_test, systems_scores_dev, args.norm)
        norm_comet_test = standardize(systems_comet_scores_test, systems_comet_scores_dev, args.norm)
        norm_comet_avg_test = norm_comet_test.mean(axis=1)
    
        # we need to repeat on the dev set to optimise the calibration parameters!
        norm_human_dev = standardize(systems_scores_dev, systems_scores_dev, args.norm)
        norm_comet_dev = standardize(systems_comet_scores_dev, systems_comet_scores_dev, args.norm)
        norm_comet_avg_dev = norm_comet_dev.mean(axis=1)    
    
        batch_range = [1]
        for batch_size in batch_range:
            batch_human_test, batch_comet_scores_test, batch_comet_avg_test, batch_comet_std_test = batch_data(
                norm_human_test, norm_comet_test, norm_comet_avg_test, batch_size=batch_size)
            batch_human_dev, batch_comet_scores_dev, batch_comet_avg_dev, batch_comet_std_dev = batch_data(
                norm_human_dev, norm_comet_dev, norm_comet_avg_dev, batch_size=batch_size)

            # Compute fixed std to use as a baseline model
            if args.baseline:
                fixed_std = compute_fixed_std(batch_comet_avg_dev, batch_human_dev)
                batch_baseline_stds_test = np.full_like(batch_comet_std_test, fixed_std)
                batch_baseline_stds_dev = np.full_like(batch_comet_std_dev, fixed_std)
                pearson_acc = stats.pearsonr(batch_comet_avg_test, batch_human_test)[0]
                base_calibration_error, gammas, base_matches = compute_calibration_error(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                mcpe_base, gammas, mcpe_matches_base = compute_mcpe(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                sharpness_base = compute_sharpness(batch_baseline_stds_test, std_sum=0, std_scale=1)
                #epiw_base, gammas, epiw_matches_base = compute_epiw(
                #    batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                mpiw_base, gammas, mpiw_matches_base = compute_mpiw(
                    batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                ence_base, ence_gammas, ence_matches_base = compute_ence(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                ence_base_rn, ence_gammas_rn, ence_matches_base_rn = compute_ence_rn(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
                ence_base_nn, ence_gammas_nn, ence_matches_base_nn = compute_ence_nn(
                    batch_human_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
        
                base_avgll, base_negll = compute_avgll(batch_human_test, batch_comet_avg_test, batch_baseline_stds_test)
                print("Baseline ALL = %f" % base_avgll)
                print("Baseline NLL = %f" % base_negll)
                print()

                cal_avgll_folds.append(base_avgll)
                calibration_error_folds.append(base_calibration_error)
                sharpness_cal_folds.append(sharpness_base)
                pearson_acc_folds.append(pearson_acc)
                ############# LATEX #################
                print('----------LATEX OUTPUTS----------')
                print('& r(da,pred)& &average NLL & ECE & Sharpness \\\\')
                print('& %.3f & 0.0 & %.3f & %.3f & %.3f  \\\\' % (pearson_acc, np.round(base_avgll, 3), np.round(base_calibration_error, 3), np.round(sharpness_base, 3)))
                
            else:
                # Parametric CE
                # It assumes a parametric Gaussian distribution for the COMET scores.
                calibration_error, gammas, matches = compute_calibration_error(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                mcpe, gammas, mcpe_matches = compute_mcpe(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1) 
                sharpness = compute_sharpness(batch_comet_std_test, std_sum=0, std_scale=1)
                #epiw, gammas, epiw_matches = compute_epiw(
                #    batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                mpiw, gammas, mpiw_matches = compute_mpiw(
                    batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                ence, ence_gammas, ence_matches = compute_ence(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                ence_rn, ence_gammas_rn, ence_matches_rn = compute_ence_rn(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
                ence_nn, ence_gammas_nn, ence_matches_nn = compute_ence_nn(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
              
                
                # Seek the best post-calibration to minimize calibration error.
                # The correction is std_transformed**2 = std_sum**2 + (std_scale*std)**2,
                # where std_sum and std_scale are correction parameters.
                std_sums = np.linspace(0, 2, 100)
                std_scales = np.linspace(1, 10, 100)
                _, std_sum, std_scale, = optimize_calibration_error(
                    batch_human_dev, batch_comet_avg_dev, batch_comet_std_dev,
                    std_sums=std_sums, std_scales=std_scales)
                
                # Compute Pearson correlation between average COMET and DA.
                pearson_acc = stats.pearsonr(batch_comet_avg_test, batch_human_test)[0]
                # print("Pearson (COMET, MQM) batch size =%d - r= %f" % (
                #     batch_size, pearson_acc))

                # Compute Pearson correlation between |COMET - DA| and COMET_std
                abs_diff = [abs(da - cs) for da, cs in zip(batch_human_test, batch_comet_avg_test)]
                pearson_d1 = stats.pearsonr(abs_diff, batch_comet_std_test)[0]
                # print("Pearson (|COMET-MQM|, COMET_std) batch size =%d - r= %f" % (
                #     batch_size, pearson_d1))
                
                # Compute Pearson correlation between |COMET - DA| and COMET_std
                abs_diff_sq = [(da - cs)**2 for da, cs in zip(batch_human_test, batch_comet_avg_test)]
                batch_comet_std_test_sq = [ x**2 for x in batch_comet_std_test]
                pearson_d2 = stats.pearsonr(abs_diff_sq, batch_comet_std_test_sq)[0]
                # print("Pearson ((COMET-MQM)**2, COMET_std**2) batch size =%d - r= %f" % (
                #     batch_size, pearson_d2))

                ## Calibrated pearsons
                batch_comet_std_test_transformed = np.sqrt(std_sum**2 + (std_scale*batch_comet_std_test)**2)
                # Compute Pearson correlation between |COMET - DA| and COMET_std
                abs_diff = [abs(da - cs) for da, cs in zip(batch_human_test, batch_comet_avg_test)]
                pearson_d1_cal = stats.pearsonr(abs_diff, batch_comet_std_test_transformed)[0]
                # print("Calibrated Pearson (|COMET-MQM|, COMET_std) batch size =%d - r= %f" % (
                #     batch_size, pearson_d1_cal))
                
                # Compute Pearson correlation between |COMET - DA| and COMET_std
                abs_diff_sq = [(da - cs)**2 for da, cs in zip(batch_human_test, batch_comet_avg_test)]
                batch_comet_std_test_transformed_sq = [ x**2 for x in batch_comet_std_test_transformed]
                pearson_d2_cal = stats.pearsonr(abs_diff_sq, batch_comet_std_test_transformed_sq)[0]
                # print("Calibrated Pearson ((COMET-MQM)**2, COMET_std**2) batch size =%d - r= %f" % (
                #     batch_size, pearson_d2_cal ))

                calibration_error, gammas, matches_cal = compute_calibration_error(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test,
                    std_sum=std_sum, std_scale=std_scale)
                
                mcpe_cal, gammas, mcpe_matches_cal = compute_mcpe(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                sharpness_cal = compute_sharpness(batch_comet_std_test, std_sum, std_scale)
                #epiw_cal, gammas, epiw_matches_cal = compute_epiw(
                #    batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                mpiw_cal, gammas, mpiw_matches_cal = compute_mpiw(
                    batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                ence_cal, ence_gammas, ence_matches_cal = compute_ence(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                ence_cal_rn, ence_gammas_rn, ence_matches_cal_rn = compute_ence_rn(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
                ence_cal_nn, ence_gammas_nn, ence_matches_cal_nn = compute_ence_nn(
                    batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)

                 
                # Compute ALL and NLL
                avgll, negll = compute_avgll(batch_human_test, batch_comet_avg_test, batch_comet_std_test)       
                # print("ALL = %f" % avgll)
                # print("NLL = %f" % negll)
                # Compute ALL and NLL
                cal_avgll, cal_negll = compute_avgll(batch_human_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)       
                print("Calibrated ALL = %f" % cal_avgll)
                print("Calibrated NLL = %f" % cal_negll)

                ############# LATEX #################
                print('----------LATEX OUTPUTS----------')
                print('& average NLL & ECE & Sharpness \\\\')
                print('& %f & %f & %f  \\\\' % (np.round(cal_avgll, 3), np.round(calibration_error, 3), np.round(sharpness_cal, 3)))
                print('& r(human, pred) & r(|pred-human|,std) \\\\')
                print('& %f & %f   \\\\' % (np.round(pearson_acc,3), np.round(pearson_d1_cal,3)))



                ################### NON PRARAMETRIC ##################
                # Compute calibration error by binning different confidence intervals.
                # Non-parametric CE
                np_calibration_error, np_gammas, np_matches = compute_calibration_error_non_parametric(
                    batch_human_test, batch_comet_scores_test)
                print("Non-parametric CE = %f" % np_calibration_error)
            
                cal_avgll_folds.append(cal_avgll)
                calibration_error_folds.append(calibration_error)
                sharpness_cal_folds.append(sharpness_cal)
                pearson_acc_folds.append(pearson_acc)
                pearson_d1_cal_folds.append(pearson_d1_cal)

    if args.baseline:
        print()
        print('------averaged over k folds------')
        print('----------LATEX OUTPUTS----------')
        print('& average NLL & ECE & Sharpness \\\\')
        print('& %.3f & 0.0 & %.3f & %.3f & %.3f  \\\\' % (np.mean(pearson_acc_folds), np.mean(cal_avgll_folds), np.mean(calibration_error_folds), np.mean(sharpness_cal_folds)))
    else:
        print()
        print(cal_avgll_folds)
        print('------averaged over k folds------')
        print('----------LATEX OUTPUTS----------')
        print('& average NLL & ECE & Sharpness \\\\')
        print('& %f & %f & %f  \\\\' % (np.mean(cal_avgll_folds), np.mean(calibration_error_folds), np.mean(sharpness_cal_folds)))
        print('& r(human, pred) & r(|pred-human|,std) \\\\')
        print('& %f & %f   \\\\' % (np.mean(pearson_acc_folds), np.mean(pearson_d1_cal_folds)))

        print()
        print(cal_avgll_folds)
        print('------averaged over k folds------')
        print('----------LATEX OUTPUTS long rounded----------')
        print('& r(human, pred) & r(|pred-human|,std) & average NLL & ECE & Sharpness \\\\')
        print('& %.3f & %.3f & %.3f & %.3f & %.3f  \\\\' % (np.mean(pearson_acc_folds), np.mean(pearson_d1_cal_folds), np.mean(cal_avgll_folds), np.mean(calibration_error_folds), np.mean(sharpness_cal_folds)))
  

   
