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
from normalisation import compute_z_norm, compute_fixed_std

def get_df(comet_dir, da_dir):
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

        system_name = '_'.join(s.split('.')[0].split('_')[1:])
        lines = [[i['src'], i['mt'], i['ref'], i['dp_runs_scores']] for i in data if 'dp_runs_scores' in i.keys()]
        df_ = pd.DataFrame(data=np.array(lines), columns=['src','mt', 'ref', 'dp_runs_scores'])
        da_scores_ = da_scores[da_scores.system == system_name]
        df = df_.merge(da_scores_, how='inner', on=['src', 'mt'])

        df['predicted_score_mean'] = df['dp_runs_scores'].apply(lambda x: np.mean(x)) # segment-level
        df['predicted_score_std'] = df['dp_runs_scores'].apply(lambda x: np.std(x)) # segment-level
        df['q-mu'] = np.abs(df['z_score'] - df['predicted_score_mean'])
        df.drop(['ref_x', 'ref_y', 'lp', 'raw_score', 'annotators'], axis=1, inplace=True)
        dfs.append(df)
        df_all = pd.concat(dfs)
        df_all.reset_index(inplace=True)
   
    return df_all

def map_psqm(psqm_file, df):
    # file format: Human-B.0       independent.281139      1       1       rater2  Michael Jackson wore tape on his nose to get front pages, former bodyguard claims       Ehemaliger Bodyguard behauptet, Michael Jackson trug Pflaster auf der Nase, um in die Presse zu kommen  4
    # file format: System   doc_name    system_id?   ?  annot# src mt  score
    scores = [-1.0]*len(df)
    print(df.head())
    df['psqm_score'] = scores

    with open(psqm_file, 'r') as psqmf:
        for line in tqdm(psqmf):
            fields = line.split('\t')
            system = fields[0].split('.')[0]
            src = fields[5]
            score = fields[7]
            idx_to_change = df.index[(df['system'] == system) & (df['src'] == src)]
            df.loc[idx_to_change,'psqm_score'] = float(score)
    print(len(df))
    new_dataframe = df[df['psqm_score'] >=0]
    print(len(new_dataframe))
    return new_dataframe


def load_mqm_scores_from_df(mqm_df):
    df = mqm_df
    print(df.head())
    systems_comet_scores = {}
    systems_mqm_scores = {}
    systems_ext = []
    for i, row in df.iterrows():
        if row['psqm_score']>=0.0:
            sent_id = int(row['index'])
            system = row['system']
            system_ext = 'system_'+system+'.json'
            if not system_ext in systems_ext:
                systems_ext.append(system_ext)
            if not system_ext in systems_comet_scores:
                systems_comet_scores[system_ext]={}
            if not system_ext in systems_mqm_scores:
                systems_mqm_scores[system_ext]={}
            scores = row['dp_runs_scores']
            systems_comet_scores[system_ext][sent_id] = scores
            systems_mqm_scores[system_ext][sent_id] = row['psqm_score']
    return(systems_comet_scores, systems_mqm_scores, systems_ext)


def load_da_scores_from_df(comet_dir, da_dir):
    df = get_df(comet_dir, da_dir)
    systems_comet_scores = {}
    systems_da_scores = {}
    systems_ext = []
    for i, row in df.iterrows():
        sent_id = int(row['index'])
        system = row['system']
        system_ext = 'system_'+system+'.json'
        if not system_ext in systems_ext:
            systems_ext.append(system_ext)
        if not system_ext in systems_comet_scores:
            systems_comet_scores[system_ext]={}
        if not system_ext in systems_da_scores:
            systems_da_scores[system_ext]={}
        scores = row['dp_runs_scores']
        systems_comet_scores[system_ext][sent_id] = scores
        systems_da_scores[system_ext][sent_id] = row['z_score']
        
    return(systems_comet_scores, systems_da_scores, systems_ext)


def split_dev_test(comet_scores, scores, systems_list):
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

        dev_len = len(comet_sys_scores)//2
        count = 0
        for comet_sent_id, score_sent_id in zip(comet_sys_scores, sys_scores):
            
            if len(comet_scores_dev[system])< dev_len:
                comet_scores_dev[system][comet_sent_id]=comet_sys_scores[comet_sent_id]
                scores_dev[system][score_sent_id]=sys_scores[score_sent_id]
            else:
                comet_scores_test[system][comet_sent_id]=comet_sys_scores[comet_sent_id]
                scores_test[system][score_sent_id]=sys_scores[score_sent_id]
        
    return(comet_scores_test, scores_test, comet_scores_dev,scores_dev)
        


def batch_data(all_da, all_comet, all_comet_avg, batch_size=1):
    
    n = len(all_da) - (len(all_da) % batch_size)
    
    batch_da = all_da[:n].reshape(n//batch_size, batch_size).mean(axis=1)
    batch_comet_scores = [i.mean(axis=0) for i in all_comet[:n, :].reshape(n//batch_size, batch_size, -1)]
    batch_comet_avg = all_comet_avg[:n].reshape(
        n//batch_size, batch_size).mean(axis=1)
    batch_comet_std = all_comet[:n, :].reshape(
        n//batch_size, batch_size, -1).mean(axis=1).std(axis=-1)
    return batch_da, batch_comet_scores, batch_comet_avg, batch_comet_std


def standardize(scores_test, scores_dev, norm):
    norm_mean = 0.0
    norm_std = 1.0
    if norm:
        norm_mean, norm_std = compute_z_norm(scores_dev)
    all_scores = np.array([val for (k,v) in scores_test.items() for (key,val) in v.items() ])
    all_scores -= norm_mean
    all_scores /= norm_std
    
    return all_scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process comet outputs')
    parser.add_argument('--comet-setup-test', type=str, 
                        help='path to comet setup to test on')
    parser.add_argument('--da-scores-test', type=str, 
                        help='path to da scores to test on')
    parser.add_argument('--psqm-file', type=str, 
                        help='path to msqm scores to test on')
    parser.add_argument('--norm', type=bool, default=True)

    args = parser.parse_args()
    

    test_year='2020'
    if '2019' in args.comet_setup_test:
        test_year='2019'
    
    systems_comet_scores, systems_da_scores, systems_ext = load_da_scores_from_df(args.comet_setup_test, args.da_scores_test)
    combined_df = get_df(args.comet_setup_test, args.da_scores_test)
    mqm_comet_df = map_psqm(args.psqm_file, combined_df)
    systems_comet_scores_mqm, systems_mqm_scores, systems_ext_mqm = load_mqm_scores_from_df(mqm_comet_df)
    
    systems_comet_scores_mqm_test, systems_mqm_scores_test, systems_comet_scores_mqm_dev, systems_mqm_scores_dev = split_dev_test(
        systems_comet_scores_mqm, systems_mqm_scores, systems_ext_mqm)

    norm_mqm_test = standardize(systems_mqm_scores_test, systems_mqm_scores_dev, args.norm)
    norm_comet_test = standardize(systems_comet_scores_mqm_test, systems_comet_scores_mqm_dev, args.norm)
    norm_comet_avg_test = norm_comet_test.mean(axis=1)
    # we need to repeat on the dev set to optimise the calibration parameters!
    norm_mqm_dev = standardize(systems_mqm_scores_dev, systems_mqm_scores_dev, args.norm)
    norm_comet_dev = standardize(systems_comet_scores_mqm_dev, systems_comet_scores_mqm_dev, args.norm)
    norm_comet_avg_dev = norm_comet_dev.mean(axis=1)    


    for batch_size in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        batch_mqm_test, batch_comet_scores_test, batch_comet_avg_test, batch_comet_std_test = batch_data(
            norm_mqm_test, norm_comet_test, norm_comet_avg_test, batch_size=batch_size)
        batch_mqm_dev, batch_comet_scores_dev, batch_comet_avg_dev, batch_comet_std_dev = batch_data(
            norm_mqm_dev, norm_comet_dev, norm_comet_avg_dev, batch_size=batch_size)

        # Compute fixed std to use as a baseline model
        fixed_std = compute_fixed_std(batch_comet_avg_dev, batch_mqm_dev)
        batch_baseline_stds_test = np.full_like(batch_comet_std_test, fixed_std)
        batch_baseline_stds_dev = np.full_like(batch_comet_std_dev, fixed_std)

        # Compute Pearson correlation between average COMET and DA.
        print("Pearson (COMET, MQM) batch size =%d - r= %f" % (
            batch_size, stats.pearsonr(batch_comet_avg_test, batch_mqm_test)[0]))

        # Compute Pearson correlation between |COMET - DA| and COMET_std
        abs_diff = [abs(da - cs) for da, cs in zip(batch_mqm_test, batch_comet_avg_test)]
        print("Pearson (|COMET-MQM|, COMET_std) batch size =%d - r= %f" % (
            batch_size, stats.pearsonr(abs_diff, batch_comet_std_test)[0]))
        # Compute calibration error by binning different confidence intervals.
        # Non-parametric CE
        calibration_error, gammas, matches = compute_calibration_error_non_parametric(
            batch_mqm_test, batch_comet_scores_test)
        print("Non-parametric CE = %f" % calibration_error)
        # Best non-parametric CE 
        scaling_vals = np.linspace(0.05, 1, 20)
        _, best_scale_val = optimize_calibration_error_non_parametric(
            batch_mqm_dev, batch_comet_scores_dev, scaling_vals=scaling_vals)
        calibration_error, gammas, matches = compute_calibration_error_non_parametric(
            batch_mqm_test, batch_comet_scores_test, scaling_val=best_scale_val)
        print("Non-parametric CE = %f (calibrated, best_scaling_val=%f)" %
              (calibration_error, best_scale_val))

        # Parametric CE
        # It assumes a parametric Gaussian distribution for the COMET scores.
        calibration_error, gammas, matches = compute_calibration_error(
            batch_mqm_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
        mcpe, gammas, mcpe_matches = compute_mcpe(
            batch_mqm_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1) 
        sharpness = compute_sharpness(batch_comet_std_test, std_sum=0, std_scale=1)
        epiw, gammas, epiw_matches = compute_epiw(
            batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
        mpiw, gammas, mpiw_matches = compute_mpiw(
            batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
        ence, ence_gammas, ence_matches = compute_ence(
            batch_mqm_test, batch_comet_avg_test, batch_comet_std_test, std_sum=0, std_scale=1)
        print("ECE = %f" % calibration_error)
        print("MCE = %f" % mcpe)
        print("Sharpness = %f" % sharpness)
        print("EPIW = %f" % epiw)
        print("MPIW = %f" % mpiw)
        print("ENCE = %f" % ence)
        print("Parametric CE  = %f" % calibration_error)
        
        # Seek the best post-calibration to minimize calibration error.
        # The correction is std_transformed**2 = std_sum**2 + (std_scale*std)**2,
        # where std_sum and std_scale are correction parameters.
        std_sums = np.linspace(0, 2, 100)
        std_scales = np.linspace(1, 10, 100)
        _, std_sum, std_scale, = optimize_calibration_error(
            batch_mqm_dev, batch_comet_avg_dev, batch_comet_std_dev,
            std_sums=std_sums, std_scales=std_scales)
        calibration_error, gammas, matches_cal = compute_calibration_error(
            batch_mqm_test, batch_comet_avg_test, batch_comet_std_test,
            std_sum=std_sum, std_scale=std_scale)
        
        mcpe_cal, gammas, mcpe_matches_cal = compute_mcpe(
            batch_mqm_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
        sharpness_cal = compute_sharpness(batch_comet_std_test, std_sum, std_scale)
        epiw_cal, gammas, epiw_matches_cal = compute_epiw(
            batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
        mpiw_cal, gammas, mpiw_matches_cal = compute_mpiw(
            batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)
        ence_cal, ence_gammas, ence_matches_cal = compute_ence(
            batch_mqm_test, batch_comet_avg_test, batch_comet_std_test, std_sum, std_scale)

        print("Calibrated ECE = %f (calibrated std_sum=%f, std_scale=%f)"  % (calibration_error, std_sum, std_scale))
        print("Calibrated MCE = %f (calibrated std_sum=%f, std_scale=%f)"  % (mcpe_cal, std_sum, std_scale))
        print("Calibrated sharpness  = %f (calibrated std_sum=%f, std_scale=%f)"  % (sharpness_cal, std_sum, std_scale))
        print("Calibrated epiw sharpness  = %f (calibrated std_sum=%f, std_scale=%f)"  % (epiw_cal, std_sum, std_scale))
        print("Calibrated mpiw sharpness = %f (calibrated std_sum=%f, std_scale=%f)"  % (mpiw_cal,  std_sum, std_scale))
        print("Calibrated ENCE = %f (calibrated std_sum=%f, std_scale=%f)"  % (ence_cal,  std_sum, std_scale))
        print("Calibrated Parametric CE = %f (calibrated std_sum=%f, std_scale=%f)" %
              (calibration_error, std_sum, std_scale))
        #####
        ## Compare to baseline
        base_calibration_error, gammas, base_matches = compute_calibration_error(
            batch_mqm_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
        mcpe_base, gammas, mcpe_matches_base = compute_mcpe(
            batch_mqm_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
        sharpness_base = compute_sharpness(batch_baseline_stds_test, std_sum=0, std_scale=1)
        epiw_base, gammas, epiw_matches_base = compute_epiw(
            batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
        mpiw_base, gammas, mpiw_matches_base = compute_mpiw(
            batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
        ence_base, ence_gammas, ence_matches_base = compute_ence(
            batch_mqm_test, batch_comet_avg_test, batch_baseline_stds_test, std_sum=0, std_scale=1)
        print("Baseline ECE = %f (baseline std = %f)"  % (base_calibration_error, fixed_std))
        print("Baseline MCE = %f (baseline std = %f)"  % (mcpe_base, fixed_std))
        print("Baseline sharpness  = %f (baseline std = %f)"  % (sharpness_base, fixed_std))
        print("Baseline epiw sharpness  = %f (baseline std = %f)"  % (epiw_base, fixed_std))
        print("Baseline mpiw sharpness = %f (baseline std = %f)"  % (mpiw_base, fixed_std))
        print("Baseline ENCE = %f (baseline std = %f)"  % (ence_base, fixed_std))
        print("Baseline Parametric CE = %f (baseline std = %f)"  % (base_calibration_error, fixed_std))

        # Compute ALL and NLL
        avgll, negll = compute_avgll(batch_mqm_test, batch_comet_avg_test, batch_comet_std_test)       
        print("ALL = %f" % avgll)
        print("NLL = %f" % negll)
        # Compute Baseline ALL and NLL
        base_avgll, base_negll = compute_avgll(batch_mqm_test, batch_comet_avg_test, batch_baseline_stds_test)
        print("Baseline ALL = %f" % base_avgll)
        print("Baseline NLL = %f" % base_negll)
        print()


        plt.xlabel('Confidence level $\gamma$')
        plt.ylabel('ECE')
        plt.title('MQM: 1719 on '+test_year+' - Batch size = %d' % batch_size)
        plt.plot(gammas, matches, 'b', label="Original ECE")
        plt.plot(gammas, matches_cal, 'r', label="Calibrated ECE")
        plt.plot(gammas, base_matches, 'g', label="Baseline ECE")
        #plt.plot(gammas, mcpe_matches, 'b:', label="Original MPE")
        #plt.plot(gammas, mcpe_matches_cal, 'r:', label="Calibrated MPE")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.show()
        plt.savefig('figures/MQM_1719-'+test_year+'-ECE_bs_'+str(batch_size)+'.png')
        plt.close()

        plt.xlabel('Confidence level $\gamma$')
        plt.ylabel('Sharpness')
        plt.title('MQM: 1719 on '+test_year+' - Batch size = %d' % batch_size)
        plt.plot(gammas, epiw_matches, 'b', label="Original sharpness")
        plt.plot(gammas, epiw_matches_cal, 'r', label="Calibrated sharpness")
        plt.plot(gammas, epiw_matches_base, 'g', label="Baseline sharpness")
        plt.plot(gammas, mpiw_matches, 'b:', label="Original max sharpness")
        plt.plot(gammas, mpiw_matches_cal, 'r:', label="Calibrated max sharpness")
        #plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.show()
        plt.savefig('figures/MQM_1719-'+test_year+'-SHARP_bs_'+str(batch_size)+'.png')
        plt.close()
       
        plt.xlabel('Bins (ascending std values) $')
        plt.ylabel('ENCE')
        plt.title('MQM: 1719 on '+test_year+' - Batch size = %d' % batch_size)
        plt.plot(gammas, ence_matches, 'b', label="Original ENCE")
        plt.plot(gammas, ence_matches_cal, 'r', label="Calibrated ENCE")
        plt.plot(gammas, ence_matches_base, 'g', label="Baseline ENCE")
        #plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.show()
        plt.savefig('figures/MQM_1719-'+test_year+'-ENCE_bs_'+str(batch_size)+'.png')
        plt.close()


